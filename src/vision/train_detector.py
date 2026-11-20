"""Fine-tune ResNet-50 or ViT-B/16 as a camera-vs-AI detector on GenImage (F5).

What it does
    Trains one detector (last 4 blocks + head unfrozen, see ``detectors.py``) for one
    seen/unseen setting with the F5 augmentation, 3 epochs, AdamW + cosine schedule,
    mixed precision on GPU. After every epoch it evaluates on the setting's test
    generators (GenImage validation split), logs to W&B and saves a checkpoint, so a
    disconnect only loses the current epoch (``--resume`` continues).

Inputs
    --root: folder that contains ``data/processed/vision/genimage/`` (e.g. the Drive
    folder ``/content/drive/MyDrive/twincheck`` or a local copy in ``/content``).

Outputs
    --ckpt-dir/<arch>_<setting>_s<seed>.pt      weights + epoch (keep on Drive; not in git, >100 MB for ViT)
    results/F5/train_<arch>_<setting>_s<seed>.json   final validation metrics

How to run (GPU)
    python -m src.vision.train_detector --arch resnet50 --setting seen --root /content/genimage_root
    python -m src.vision.train_detector --arch vit_b16 --setting unseen_sd --root ... --resume
    Quick CPU check: --limit 8 --epochs 1 --batch 4 --device cpu
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.vision.detectors import (GenImageDataset, build_model, eval_transform, manifest_rows,
                                  train_transform)
from src.vision.tracking import finish, log_summary, start_run


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """P(AI) and labels for every image in the loader."""
    model.eval()
    probs, labels = [], []
    for x, y, _ in loader:
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logits = model(x.to(device)).squeeze(1)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics(p_ai: np.ndarray, y: np.ndarray, rows) -> dict:
    """Accuracy (threshold 0.5), AUROC and accuracy per generator."""
    pred = (p_ai >= 0.5).astype(int)
    out = {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
           "auroc": float(roc_auc_score(y, p_ai)) if len(set(y)) > 1 else None}
    for gen, idx in rows.groupby("generator").indices.items():
        out[f"acc_{gen}"] = float(accuracy_score(y[idx], pred[idx]))
    return out


def main() -> None:
    """Train one (arch, setting, seed) and save checkpoint + metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arch", choices=["resnet50", "vit_b16"], required=True)
    ap.add_argument("--setting", choices=["seen", "unseen_biggan", "unseen_sd"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="N images per generator (quick check)")
    ap.add_argument("--ckpt-dir", default="models/F5")
    ap.add_argument("--out", default="results/F5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    name = f"{args.arch}_{args.setting}_s{args.seed}"
    ckpt_path = Path(args.ckpt_dir) / f"{name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    train_rows = manifest_rows("train", args.setting, "train", args.limit)
    test_rows = manifest_rows("validation", args.setting, "test", args.limit)
    train_loader = DataLoader(GenImageDataset(train_rows, args.root, train_transform()),
                              batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              generator=torch.Generator().manual_seed(args.seed))
    test_loader = DataLoader(GenImageDataset(test_rows, args.root, eval_transform()),
                             batch_size=args.batch, num_workers=args.workers)

    model = build_model(args.arch).to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler(enabled=args.device == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    start_epoch = 0
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        if start_epoch >= args.epochs:
            print(f"{name} already finished ({args.epochs} epochs): skipping")
            return
        print(f"resumed from epoch {start_epoch}")

    config = {"seed": args.seed, "arch": args.arch, "setting": args.setting, "epochs": args.epochs,
              "lr": args.lr, "batch": args.batch, "unfrozen": "last 4 blocks + head",
              "augmentation": "rrc0.6, hflip, jpeg60-95, blur p0.3",
              "n_train": len(train_rows), "n_test": len(test_rows)}
    run = start_run(f"F5-{name}", config, not args.no_wandb)
    result = {}
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total, seen_n = 0.0, 0
        for step, (x, y, _) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            with torch.autocast(device_type="cuda", enabled=args.device == "cuda"):
                loss = loss_fn(model(x).squeeze(1), y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total, seen_n = total + loss.item() * len(y), seen_n + len(y)
            if run is not None and step % 50 == 0:
                run.log({"train_loss": loss.item(), "epoch": epoch})
        p_ai, y = predict(model, test_loader, args.device)
        result = {"epoch": epoch, "train_loss": total / max(1, seen_n), **metrics(p_ai, y, test_rows)}
        print(json.dumps(result))
        if run is not None:
            run.log({f"val_{k}": v for k, v in result.items() if isinstance(v, float)})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "config": config}, ckpt_path)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / f"train_{name}.json", "w") as f:
        json.dump({**config, **result, "wandb": run.url if run is not None and hasattr(run, "url") else None}, f, indent=2)
    log_summary(run, {k: v for k, v in result.items() if isinstance(v, float)})
    finish(run)


if __name__ == "__main__":
    main()
