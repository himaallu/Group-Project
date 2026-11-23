from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.vision.detectors import (CachedImages, GenImageDataset, build_model, eval_transform,
                                  freeze_batchnorm, load_cache, manifest_rows,
                                  restore_imagenet_batchnorm, train_transform)
from src.vision.tracking import finish, log_summary, start_run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y, _ in loader:
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logits = model(x.to(device)).squeeze(1)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics(p_ai: np.ndarray, y: np.ndarray, rows) -> dict:
    pred = (p_ai >= 0.5).astype(int)
    out = {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "auroc": float(roc_auc_score(y, p_ai)) if len(set(y)) > 1 else None}
    for gen, idx in rows.groupby("generator").indices.items():
        out[f"acc_{gen}"] = float(accuracy_score(y[idx], pred[idx]))
    return out


def cached_loader(blobs, labels, transform, batch: int, workers: int, device: str,
                  corruption: str = "clean", shuffle: bool = False, seed: int = 42) -> DataLoader:
    return DataLoader(CachedImages(blobs, labels, transform, corruption), batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=device == "cuda",
                      generator=torch.Generator().manual_seed(seed) if shuffle else None)


@torch.inference_mode()
def predict_cached(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    probs = []
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            probs.append(torch.sigmoid(model(x).squeeze(1).float()).cpu().numpy())
    return np.concatenate(probs)


def run_name(args) -> str:
    return f"{args.arch}_s{args.seed}" + ("_freezebn" if args.freeze_bn else "")


def save_predictions(rows: dict, probs: dict, path: Path) -> None:
    import pandas as pd
    parts = [rows[s][["md5", "split", "generator", "class_name"]].assign(p_ai=probs[s]) for s in probs]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)


def run_cached(args) -> dict:
    set_seed(args.seed)
    rows = {s: manifest_rows(s, args.limit) for s in ("train", "validation", "test")}
    blobs = {s: load_cache(args.cache, rows[s]["md5"]) for s in rows}
    labels = {s: (rows[s]["class_name"] == "fake").astype(np.float32).to_numpy() for s in rows}
    name = run_name(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(args.arch, pretrained=not args.eval_only).to(args.device, memory_format=torch.channels_last)

    def evaluate(tag: str, history: list | None = None) -> dict:
        loaders = {s: cached_loader(blobs[s], labels[s], eval_transform(), args.batch, args.workers, args.device)
                   for s in ("validation", "test")}
        probs = {s: predict_cached(model, loaders[s], args.device) for s in loaders}
        result = {"seed": args.seed, "arch": args.arch, "run": tag,
                  "split": "F1 (train/val real + BigGAN, test real + SD v1.4)",
                  "input": "F3 preprocessing: centre crop, 96 px, JPEG with real-photo settings, greyscale share matched",
                  "epochs": args.epochs, "lr": args.lr, "batch": args.batch, "unfrozen": "last 4 blocks + head",
                  "augmentation": "rrc0.6, hflip, jpeg60-95, blur p0.3", "mixed_precision": "fp16",
                  "freeze_bn": bool(args.freeze_bn), "restore_imagenet_bn": bool(args.restore_imagenet_bn),
                  "n_train": len(rows["train"]), "n_val": len(rows["validation"]), "n_test": len(rows["test"])}
        if history is not None:
            result["history"] = history
        for s in loaders:
            result[s] = metrics(probs[s], labels[s], rows[s])
        save_predictions(rows, probs, Path(args.preds_dir) / f"preds_{tag}.csv")
        return result

    if args.eval_only:
        state = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(state["model"])
        tag = name + ("_imagenetbn" if args.restore_imagenet_bn else "_check")
        if args.restore_imagenet_bn:
            restore_imagenet_batchnorm(model)
        result = evaluate(tag)
        with open(out / f"eval_{tag}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps({s: result[s] for s in ("validation", "test")}), flush=True)
        return result

    ckpt_path = Path(args.ckpt_dir) / f"{name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    train_dl = cached_loader(blobs["train"], labels["train"], train_transform(), args.batch, args.workers,
                             args.device, shuffle=True, seed=args.seed)
    val_dl = cached_loader(blobs["validation"], labels["validation"], eval_transform(), args.batch, args.workers,
                           args.device)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler(enabled=args.device == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    start, history = 0, []
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, history = state["epoch"] + 1, state["history"]
        print(f"{name}: continuing after epoch {start}", flush=True)
    for epoch in range(start, args.epochs):
        began = time.time()
        model.train()
        if args.freeze_bn:
            freeze_batchnorm(model)
        total = seen = 0
        for x, y, _ in train_dl:
            x = x.to(args.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device == "cuda"):
                loss = loss_fn(model(x).squeeze(1), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total, seen = total + loss.item() * len(y), seen + len(y)
        val = metrics(predict_cached(model, val_dl, args.device), labels["validation"], rows["validation"])
        history.append({"epoch": epoch + 1, "train_loss": total / seen, "val_accuracy": val["accuracy"],
                        "seconds": round(time.time() - began, 1)})
        print(json.dumps({"run": name, **history[-1]}), flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history}, ckpt_path)
    result = evaluate(name, history)
    with open(out / f"train_{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({s: result[s] for s in ("validation", "test")}), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet50", "vit_b16"], required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ckpt-dir", default="models/F5")
    ap.add_argument("--out", default="results/F5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--freeze-bn", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--restore-imagenet-bn", action="store_true")
    ap.add_argument("--preds-dir", default="results/F10")
    args = ap.parse_args()
    if args.cache:
        run_cached(args)
        return
    if not args.root:
        ap.error("--root is required without --cache")
    set_seed(args.seed)
    name = f"{args.arch}_s{args.seed}"
    ckpt_path = Path(args.ckpt_dir) / f"{name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    train_rows = manifest_rows("train", args.limit)
    val_rows = manifest_rows("validation", args.limit)
    test_rows = manifest_rows("test", args.limit)
    train_loader = DataLoader(GenImageDataset(train_rows, args.root, train_transform()),
                              batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(GenImageDataset(val_rows, args.root, eval_transform()),
                            batch_size=args.batch, num_workers=args.workers)
    test_loader = DataLoader(GenImageDataset(test_rows, args.root, eval_transform()),
                             batch_size=args.batch, num_workers=args.workers)

    model = build_model(args.arch).to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler(enabled=args.device == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    start_epoch, train_loss, wandb_url = 0, None, None
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, train_loss, wandb_url = state["epoch"] + 1, state.get("train_loss"), state.get("wandb")
        print(f"resumed from epoch {start_epoch}")

    config = {"seed": args.seed, "arch": args.arch, "split": "F1 (train/val real + BigGAN, test real + SD v1.4)",
              "epochs": args.epochs, "lr": args.lr, "batch": args.batch, "unfrozen": "last 4 blocks + head",
              "augmentation": "rrc0.6, hflip, jpeg60-95, blur p0.3",
              "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows)}
    run = start_run(f"F5-{name}", config, not args.no_wandb and start_epoch < args.epochs)
    if run is not None and getattr(run, "url", None):
        wandb_url = run.url
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
        train_loss = total / max(1, seen_n)
        p_ai, y = predict(model, val_loader, args.device)
        val = metrics(p_ai, y, val_rows)
        print(json.dumps({"epoch": epoch, "train_loss": train_loss, **val}))
        if run is not None:
            run.log({f"val_{k}": v for k, v in val.items() if isinstance(v, float)})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "train_loss": train_loss,
                    "wandb": wandb_url, "config": config}, ckpt_path)

    result = {"epochs_trained": args.epochs, "train_loss": train_loss}
    for split, loader, rows in (("validation", val_loader, val_rows), ("test", test_loader, test_rows)):
        p_ai, y = predict(model, loader, args.device)
        result[split] = metrics(p_ai, y, rows)
    print(json.dumps(result))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / f"train_{name}.json", "w") as f:
        json.dump({**config, **result, "wandb": wandb_url}, f, indent=2)
    log_summary(run, {f"{split}_{k}": v for split in ("validation", "test")
                      for k, v in result[split].items() if isinstance(v, float)})
    finish(run)


if __name__ == "__main__":
    main()
