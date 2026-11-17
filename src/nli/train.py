"""Fine-tune a multilingual NLI model on the weak TwinCheck pairs (R4).

What it does
    1. Loads the weak training pairs (``build_pairs.py``), the weak val pairs and the
       hand-labelled gold pairs (``data/nli/gold_val.jsonl``, H2 + Y2).
    2. Scores gold + weak val with the model as it is (zero-shot).
    3. Fine-tunes it: cross-entropy over entail / neutral / contradict, AdamW lr 2e-5,
       batch 16, max length 256, 2 epochs, 10% linear warmup, seed 42. Mixed precision
       is bf16 on GPUs that support it (A100, L4) and fp16 otherwise: DeBERTa-v3 can
       overflow in fp16, bf16 cannot. Loss goes to W&B (project ``twincheck``).
    4. Scores gold + weak val again and saves the model and the before/after numbers:
       macro-F1, per-class precision / recall / F1 and the confusion matrix, for
       English and Arabic gold separately.
    Input order is the same as ``src/nli/score_zeroshot.py``: premise = snippet,
    hypothesis = claim, and only the snippet is cut when the pair is too long.
    A checkpoint is written every ~20 minutes; rerunning the script resumes from it.

Inputs
    --train data/nli/train.jsonl   --val data/nli/val_weak.jsonl   --gold data/nli/gold_val.jsonl

Outputs
    --out       models/nli_v1/   (model + tokenizer; uploaded to the HF Hub by the notebook)
    --metrics   results/R4/gold_eval.json
    --ckpt-dir  /content/ckpt_nli/  (resume checkpoint; not in git)

How to run (Colab GPU, repo root)
    python -m src.nli.train --n-train 30 --out models/test10/nli \
        --metrics models/test10/eval.json --no-wandb                 # 10-row check
    python -m src.nli.train                                          # mDeBERTa (main)
    python -m src.nli.train --model joeddav/xlm-roberta-large-xnli --lr 1e-5 \
        --out models/nli_xlmr --metrics results/R4/gold_eval_xlmr.json   # comparison
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from src.eval.loaders import load_jsonl
from src.nli.score_zeroshot import DEFAULT_MODEL, label_columns

CLASSES = ("entail", "contradict", "neutral")
COLUMN_OF = {"entail": "p_entail", "contradict": "p_contradict", "neutral": "p_neutral"}


def set_seed(seed: int) -> None:
    """Fix the Python, NumPy and torch random seeds."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_report(gold: list[str], pred: list[str]) -> dict:
    """Macro-F1, per-class precision / recall / F1 / support and the confusion matrix.

    A class that never occurs in ``gold`` and is never predicted is left out of the
    macro average (so a gold set without Arabic entail pairs is not punished for it).
    """
    per_class, f1s = {}, []
    for c in CLASSES:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        n_pred, n_gold = pred.count(c), gold.count(c)
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_gold if n_gold else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": n_gold}
        if n_gold or n_pred:
            f1s.append(f1)
    confusion = {g: {p: sum(a == g and b == p for a, b in zip(gold, pred)) for p in CLASSES}
                 for g in CLASSES}
    return {"n": len(gold), "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
            "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4)
            if gold else None,
            "per_class": per_class, "confusion (rows = gold)": confusion}


def encode(tokenizer, pairs: list[dict], max_len: int, device: str):
    """Tokenize (snippet, claim) pairs; only the snippet is truncated."""
    return tokenizer([p["snippet"] for p in pairs], [p["claim"] for p in pairs],
                     truncation="only_first", max_length=max_len, padding=True,
                     return_tensors="pt").to(device)


def predict(model, tokenizer, pairs: list[dict], max_len: int, device: str, dtype,
            batch_size: int = 64) -> list[str]:
    """Predicted class name (entail / contradict / neutral) for every pair."""
    import torch

    index_of = label_columns(model.config.id2label)          # p_entail -> model output index
    name_of = {index_of[COLUMN_OF[c]]: c for c in CLASSES}
    model.eval()
    preds = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype,
                                         enabled=device == "cuda"):
        for start in range(0, len(pairs), batch_size):
            logits = model(**encode(tokenizer, pairs[start:start + batch_size],
                                    max_len, device)).logits
            preds.extend(name_of[i] for i in logits.argmax(-1).cpu().tolist())
    model.train()
    return preds


def evaluate(model, tokenizer, gold: list[dict], val: list[dict], max_len: int,
             device: str, dtype) -> dict:
    """Class reports on gold (per language) and on the weak val pairs."""
    result = {}
    for lang in sorted({p.get("language", "en") for p in gold}):
        rows = [p for p in gold if p.get("language", "en") == lang]
        result[f"gold_{lang}"] = class_report(
            [p["label"] for p in rows], predict(model, tokenizer, rows, max_len, device, dtype))
    if val:
        result["val_weak"] = class_report(
            [p["label"] for p in val], predict(model, tokenizer, val, max_len, device, dtype))
    return result


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, step: int) -> None:
    """Save everything needed to resume training at ``step`` (atomic write)."""
    import torch

    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "state.pt.tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "step": step}, tmp)
    os.replace(tmp, path / "state.pt")
    print(f"  checkpoint at step {step} -> {path}")


def train(args: argparse.Namespace) -> dict:
    """Run the fine-tuning job described in the module docstring.

    Returns:
        The metrics dict that is also written to ``args.metrics``.
    """
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    train_pairs = load_jsonl(args.train)
    if args.n_train:
        train_pairs = train_pairs[:args.n_train]
    val = load_jsonl(args.val)[:args.n_val] if Path(args.val).exists() else []
    gold = load_jsonl(args.gold)
    print(f"{len(train_pairs)} train, {len(val)} weak val, {len(gold)} gold pairs; "
          f"device {device}, {'bf16' if use_bf16 else 'fp16'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    index_of = label_columns(model.config.id2label)
    target = {c: index_of[COLUMN_OF[c]] for c in CLASSES}   # our label -> model output

    run = None
    if args.wandb:
        import wandb
        short = args.model.split("/")[-1]
        run = wandb.init(project="twincheck", name=f"R4-nli-{short}-seed{args.seed}",
                         config=vars(args), tags=["R4", "nli"])

    before = evaluate(model, tokenizer, gold, val, args.max_len, device, dtype)
    print("zero-shot:", json.dumps({k: v["macro_f1"] for k, v in before.items()}))

    steps_per_epoch = (len(train_pairs) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup * total_steps), total_steps)
    # A grad scaler is only needed for fp16; with bf16 (or on CPU) it does nothing.
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and not use_bf16)
    loss_fn = torch.nn.CrossEntropyLoss()

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
        random.Random(args.seed + epoch).shuffle(order)      # same order on resume
        for batch_no in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + batch_no
            if global_step < step:
                continue                                    # done before the interruption
            batch = [train_pairs[i] for i in
                     order[batch_no * args.batch_size:(batch_no + 1) * args.batch_size]]
            labels = torch.tensor([target[p["label"]] for p in batch], device=device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device == "cuda"):
                logits = model(**encode(tokenizer, batch, args.max_len, device)).logits
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
                print(f"epoch {epoch + 1} step {step}/{total_steps} loss {loss.item():.4f} "
                      f"({time.time() - start:.0f}s)")
                if run:
                    run.log({"train/loss": loss.item(), "lr": scheduler.get_last_lr()[0]},
                            step=step)
            if time.time() - last_ckpt > args.ckpt_minutes * 60:
                save_checkpoint(Path(args.ckpt_dir), model, optimizer, scheduler, scaler, step)
                last_ckpt = time.time()

    train_minutes = (time.time() - start) / 60
    after = evaluate(model, tokenizer, gold, val, args.max_len, device, dtype)
    print("fine-tuned:", json.dumps({k: v["macro_f1"] for k, v in after.items()}))

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)   # fp32: safe for DeBERTa
    tokenizer.save_pretrained(args.out)

    metrics = {"base_model": args.model, "seed": args.seed, "n_train_pairs": len(train_pairs),
               "lr": args.lr, "batch_size": args.batch_size, "max_len": args.max_len,
               "epochs": args.epochs, "warmup": args.warmup,
               "precision": "bf16" if use_bf16 else "fp16",
               "train_minutes": round(train_minutes, 1),
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               "zero_shot": before, "fine_tuned": after,
               "wandb_url": getattr(run, "url", None) if run else None}
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if run:
        for when, res in (("zero_shot", before), ("fine_tuned", after)):
            run.summary.update({f"{key}/macro_f1_{when}": v["macro_f1"]
                                for key, v in res.items()})
        run.finish()
    print(f"saved model -> {args.out}, metrics -> {args.metrics}")
    return metrics


def main() -> None:
    """Parse arguments and train."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", default="data/nli/train.jsonl")
    ap.add_argument("--val", default="data/nli/val_weak.jsonl")
    ap.add_argument("--gold", default="data/nli/gold_val.jsonl")
    ap.add_argument("--n-train", type=int, help="first N train pairs only (10-row check)")
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="models/nli_v1")
    ap.add_argument("--metrics", default="results/R4/gold_eval.json")
    ap.add_argument("--ckpt-dir", default="/content/ckpt_nli")
    ap.add_argument("--ckpt-minutes", type=float, default=20)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
