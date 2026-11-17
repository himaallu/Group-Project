"""Fine-tune the BGE reranker on (claim, snippet) relevance pairs (R2).

What it does
    1. Builds ~20k training pairs from the English train files (own snippet = 1,
       random other claim's snippet = 0; ``build_pairs.make_pairs``, seed 42) and
       ~2k validation pairs from the val files.
    2. Scores the val pairs with the base model (zero-shot accuracy + AUROC).
    3. Fine-tunes ``BAAI/bge-reranker-v2-m3`` for 1 epoch: binary cross-entropy on
       the model's single logit, AdamW lr 2e-5, batch 16, max length 256, 10% linear
       warmup, fp16, seed 42. Logs loss to W&B (project ``twincheck``).
    4. Scores the val pairs again and saves the model (fp16 safetensors) and the
       before/after numbers.
    A checkpoint is written every ~20 minutes; rerunning the script resumes from it.

Inputs
    --train-files  data/processed/en/{multifc,averitec}_train.jsonl
    --val-files    data/processed/en/{multifc,averitec}_val.jsonl

Outputs
    --out          models/reranker_v1/   (model + tokenizer)
    --metrics      results/R2/reranker_eval.json
    --ckpt-dir     /content/ckpt_reranker/  (resume checkpoint; not in git)

How to run (Colab GPU, from the repo root)
    python -m src.retrieval.train_reranker --n-train-claims 5 --n-val-claims 5 \
        --out models/test10/reranker --metrics models/test10/eval.json --no-wandb  # 10-row check
    python -m src.retrieval.train_reranker                                         # full run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from src.retrieval.build_pairs import binary_auroc, load_records, make_pairs

BASE_MODEL = "BAAI/bge-reranker-v2-m3"
EN = "data/processed/en"
TRAIN_FILES = [f"{EN}/multifc_train.jsonl", f"{EN}/averitec_train.jsonl"]
VAL_FILES = [f"{EN}/multifc_val.jsonl", f"{EN}/averitec_val.jsonl"]


def set_seed(seed: int) -> None:
    """Fix the Python, NumPy and torch random seeds."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tokenizer, pairs: list[dict], max_len: int, device: str):
    """Tokenize a batch of pairs as (claim, snippet) and move it to the device."""
    return tokenizer([p["claim"] for p in pairs], [p["snippet"] for p in pairs],
                     truncation=True, max_length=max_len, padding=True,
                     return_tensors="pt").to(device)


def evaluate(model, tokenizer, pairs: list[dict], max_len: int, device: str,
             batch_size: int = 64) -> dict:
    """Accuracy (threshold 0.5) and AUROC of the model on labelled pairs.

    Returns:
        {"n_pairs", "accuracy", "auroc", "mean_relevance_pos", "mean_relevance_neg"}
    """
    import torch

    model.eval()
    scores = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            logits = model(**encode(tokenizer, batch, max_len, device)).logits[:, 0]
            scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
    labels = [p["label"] for p in pairs]
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    return {
        "n_pairs": len(pairs),
        "accuracy": sum((s >= 0.5) == (y == 1) for s, y in zip(scores, labels)) / len(pairs),
        "auroc": binary_auroc(labels, scores),
        "mean_relevance_pos": sum(pos) / len(pos) if pos else None,
        "mean_relevance_neg": sum(neg) / len(neg) if neg else None,
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, step: int) -> None:
    """Save everything needed to resume training at ``step``."""
    import torch

    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "state.pt.tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "step": step}, tmp)
    os.replace(tmp, path / "state.pt")  # atomic: a disconnect never leaves half a file
    print(f"  checkpoint at step {step} -> {path}")


def train(args: argparse.Namespace) -> dict:
    """Run the full fine-tuning job described in the module docstring.

    Returns:
        The metrics dict that is also written to ``args.metrics``.
    """
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_pairs = make_pairs(load_records(args.train_files), args.n_train_claims, args.seed)
    val_pairs = make_pairs(load_records(args.val_files), args.n_val_claims, args.seed)
    print(f"{len(train_pairs)} train pairs, {len(val_pairs)} val pairs, device {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="twincheck", name=f"R2-reranker-seed{args.seed}",
                         config=vars(args), tags=["R2", "reranker"])

    before = evaluate(model, tokenizer, val_pairs, args.max_len, device)
    print("zero-shot:", before)

    steps_per_epoch = (len(train_pairs) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Resume from the last checkpoint if this run was interrupted.
    step, ckpt = 0, Path(args.ckpt_dir) / "state.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step = state["step"]
        print(f"resumed from step {step}")

    model.train()
    start, last_ckpt = time.time(), time.time()
    for epoch in range(args.epochs):
        order = list(range(len(train_pairs)))
        random.Random(args.seed + epoch).shuffle(order)  # same order on resume
        for batch_no in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + batch_no
            if global_step < step:
                continue  # already done before the interruption
            idx = order[batch_no * args.batch_size:(batch_no + 1) * args.batch_size]
            batch = [train_pairs[i] for i in idx]
            labels = torch.tensor([p["label"] for p in batch], dtype=torch.float, device=device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(**encode(tokenizer, batch, args.max_len, device)).logits[:, 0]
            loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step = global_step + 1
            if step % 50 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} loss {loss.item():.4f} "
                      f"({time.time() - start:.0f}s)")
                if run:
                    run.log({"train/loss": loss.item(), "lr": scheduler.get_last_lr()[0]},
                            step=step)
            if time.time() - last_ckpt > args.ckpt_minutes * 60:
                save_checkpoint(Path(args.ckpt_dir), model, optimizer, scheduler, scaler, step)
                last_ckpt = time.time()

    train_minutes = (time.time() - start) / 60
    after = evaluate(model, tokenizer, val_pairs, args.max_len, device)
    print("fine-tuned:", after)

    # fp16 weights halve the size (~1.1 GB) with no visible change in scores.
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    metrics = {"base_model": args.model, "seed": args.seed, "n_train_pairs": len(train_pairs),
               "lr": args.lr, "batch_size": args.batch_size, "max_len": args.max_len,
               "epochs": args.epochs, "train_minutes": round(train_minutes, 1),
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               "val_zero_shot": before, "val_fine_tuned": after,
               "wandb_url": run.get_url() if run else None}
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if run:
        run.summary.update({"val/acc_zero_shot": before["accuracy"],
                            "val/auroc_zero_shot": before["auroc"],
                            "val/acc": after["accuracy"], "val/auroc": after["auroc"]})
        run.finish()
    print(f"saved model -> {args.out}, metrics -> {args.metrics}")
    return metrics


def main() -> None:
    """Parse arguments and train."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-files", nargs="+", default=TRAIN_FILES)
    ap.add_argument("--val-files", nargs="+", default=VAL_FILES)
    ap.add_argument("--n-train-claims", type=int, default=10_000, help="2 pairs per claim")
    ap.add_argument("--n-val-claims", type=int, default=1_000)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--out", default="models/reranker_v1")
    ap.add_argument("--metrics", default="results/R2/reranker_eval.json")
    ap.add_argument("--ckpt-dir", default="/content/ckpt_reranker")
    ap.add_argument("--ckpt-minutes", type=float, default=20)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
