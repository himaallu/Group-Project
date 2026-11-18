"""Train the verdict head with the twin loss (H3; λ grid in H4; ablations in H5).

What it does
    1. Builds the features (``features.py``) for train / val / test originals and
       any twin files.
    2. Trains the MLP (``model.py``) with
           loss = CE_w(originals)                                   class-weighted
                + λ_inv  * KL(p_orig || p_twin)  for _P / _A / _M  (keep the answer;
                                                  the original's side is detached)
                + λ_sens * CE(_B -> contradicted, _C -> unverified)
       Adam lr 1e-3, up to 30 epochs, early stopping on validation macro-F1
       (patience 5). Each batch holds whole groups (an original with its twins).
       v0 = λ_inv = λ_sens = 0 (no twin terms).
    3. There are no validation twins (R3 makes test + train twins only), so twins of
       10% of the train sources (seed 42) are held out of training and used, with
       their originals, to report validation consistency / ESR for the λ grid.
    4. Writes predictions, validation logits (for ``calibrate.py``), metrics, the
       model, and logs to W&B project ``twincheck`` if wandb is installed.

Inputs
    --train/--val/--test  data/processed/en/multifc_{train,val,test}.jsonl
    --twins               data/copies/en/{train,test}.jsonl (optional)
    --nli                 per-snippet NLI parquet files (zero-shot or Rathi's R4),
                          covering every record and twin used
    --relevance           features/relevance_en.parquet (+ twin relevance file(s))
    --entities            features/entities_en.jsonl

Outputs (NAME = --name, SEED = --seed)
    models/verdict_NAME_sSEED/{model.pt, config.json}
    results/H3/preds_NAME_sSEED.jsonl       test originals + test twins (§5 format)
    results/H3/val_logits_NAME_sSEED.jsonl  {id, gold, logits} for calibrate.py
    results/H3/metrics_NAME_sSEED.json      test + validation metrics, settings

How to run (Colab; CPU is fine, the head is tiny)
    python -m src.verdict.train --name v0 --seed 42 \
        --nli features/nli_zeroshot/multifc_{train,val,test}.parquet
    # with twins and twin loss (H4 grid):
    python -m src.verdict.train --name inv1_sens1 --lambda-inv 1 --lambda-sens 1 \
        --twins data/copies/en/train.jsonl data/copies/en/test.jsonl --nli ...
    # ablations: --no-relevance (no reranker), --drop-entity (no entity match)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.eval.loaders import join_gold, load_jsonl, split_twin_id
from src.eval.metrics import LABELS, compute_all, macro_f1
from src.verdict.features import FEATURES, build_features, load_entities
from src.verdict.model import VerdictHead, fit_scaler, predict, save_head, standardise

EN = "data/processed/en"
INVARIANT = ("P", "A", "M")
SENSITIVE_TARGET = {"B": LABELS.index("contradicted"), "C": LABELS.index("unverified")}
ENTITY_FEATURES = ("match_place", "match_date", "match_person")


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

def class_weights(labels: list[str]) -> torch.Tensor:
    """Inverse-frequency weights n / (3 * count) in LABELS order (1.0 if a class is absent)."""
    counts = np.array([labels.count(lab) for lab in LABELS], dtype=float)
    w = np.where(counts > 0, len(labels) / (3 * np.maximum(counts, 1)), 1.0)
    return torch.tensor(w, dtype=torch.float32)


def twin_loss(logits: torch.Tensor, kinds: list[str | None], targets: torch.Tensor,
              orig_pos: torch.Tensor, weights: torch.Tensor, lambda_inv: float,
              lambda_sens: float) -> tuple[torch.Tensor, dict]:
    """Total loss for one batch of grouped rows.

    Args:
        logits: (b, 3) model outputs.
        kinds: twin type per row (None for originals).
        targets: (b,) gold label index per row (used for originals).
        orig_pos: (b,) row position of each twin's original in this batch, -1 if absent.
        weights: class weights for the originals' cross-entropy.
        lambda_inv, lambda_sens: weights of the two twin terms.

    Returns:
        (total loss, {"ce", "inv", "sens"} as floats for logging).
    """
    is_orig = torch.tensor([k is None for k in kinds])
    ce = F.cross_entropy(logits[is_orig], targets[is_orig], weight=weights) \
        if is_orig.any() else logits.sum() * 0.0

    inv = logits.sum() * 0.0
    inv_rows = [i for i, k in enumerate(kinds) if k in INVARIANT and orig_pos[i] >= 0]
    if lambda_inv > 0 and inv_rows:
        rows = torch.tensor(inv_rows)
        p_orig = F.softmax(logits[orig_pos[rows]], dim=-1).detach()  # the original is the teacher
        log_p_twin = F.log_softmax(logits[rows], dim=-1)
        inv = (p_orig * (torch.log(p_orig + 1e-12) - log_p_twin)).sum(dim=-1).mean()

    sens = logits.sum() * 0.0
    sens_rows = [i for i, k in enumerate(kinds) if k in SENSITIVE_TARGET]
    if lambda_sens > 0 and sens_rows:
        rows = torch.tensor(sens_rows)
        tgt = torch.tensor([SENSITIVE_TARGET[kinds[i]] for i in sens_rows])
        sens = F.cross_entropy(logits[rows], tgt)

    total = ce + lambda_inv * inv + lambda_sens * sens
    return total, {"ce": ce.item(), "inv": inv.item(), "sens": sens.item()}


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

def make_groups(table: pd.DataFrame) -> list[list[int]]:
    """Row indices grouped by base_id, original first (twins without an original form their own group)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, (base, kind) in enumerate(zip(table["base_id"], table["twin_type"])):
        if kind is None:
            groups[base].insert(0, i)
        else:
            groups[base].append(i)
    return list(groups.values())


def batches(groups: list[list[int]], batch_groups: int, rng: random.Random):
    """Yield lists of row indices; each batch has ``batch_groups`` whole groups."""
    order = groups[:]
    rng.shuffle(order)
    for start in range(0, len(order), batch_groups):
        yield [i for g in order[start:start + batch_groups] for i in g]


def batch_tensors(table: pd.DataFrame, rows: list[int]):
    """kinds, targets and orig_pos for the rows of one batch."""
    sub = table.iloc[rows]
    kinds = [None if pd.isna(k) else k for k in sub["twin_type"]]
    targets = torch.tensor([LABELS.index(lab) for lab in sub["label"]])
    pos_of_orig = {b: j for j, (b, k) in enumerate(zip(sub["base_id"], kinds)) if k is None}
    orig_pos = torch.tensor([pos_of_orig.get(b, -1) if k is not None else -1
                             for b, k in zip(sub["base_id"], kinds)])
    return kinds, targets, orig_pos


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def pred_rows(table: pd.DataFrame, logits: np.ndarray) -> list[dict]:
    """§5 prediction records from logits (argmax, softmax probabilities)."""
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    rows = []
    for rec_id, p in zip(table["id"], probs):
        p = np.round(p, 6)
        p = p / p.sum()
        rows.append({"id": rec_id, "pred_label": LABELS[int(p.argmax())],
                     "probs": {lab: float(v) for lab, v in zip(LABELS, p)}})
    return rows


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    """Write dicts as JSON Lines (creates the folder)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def evaluate(preds: list[dict], gold_records: list[dict]) -> dict:
    """compute_all on predictions joined with their gold records."""
    df, missing = join_gold(preds, gold_records)
    if missing:
        raise ValueError(f"{len(missing)} predictions have no gold record, e.g. {missing[:3]}")
    return compute_all(df)


def split_val_twins(twins: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out the train twins of ``frac`` of the train sources (by base id).

    Returns:
        (train twins kept for training, held-out validation twins).
    """
    train_tw = [t for t in twins if t["split"] == "train"]
    bases = sorted({split_twin_id(t["id"])[0] for t in train_tw})
    held = set(random.Random(seed).sample(bases, int(round(frac * len(bases))))) if bases else set()
    keep = [t for t in train_tw if split_twin_id(t["id"])[0] not in held]
    val = [t for t in train_tw if split_twin_id(t["id"])[0] in held]
    return keep, val


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(args: argparse.Namespace) -> dict:
    """Full run (see module docstring). Returns the metrics dict that is also saved."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    load = lambda paths: [r for p in paths for r in load_jsonl(p)]
    train_recs, val_recs, test_recs = load(args.train), load(args.val), load(args.test)
    if args.limit:  # quick check on a few records
        train_recs, val_recs, test_recs = (r[:args.limit] for r in (train_recs, val_recs, test_recs))
    twins = load(args.twins) if args.twins else []
    train_tw, val_tw = split_val_twins(twins, args.val_twin_frac, args.seed)
    test_tw = [t for t in twins if t["split"] == "test"]
    if any(t["split"] == "val" for t in twins):
        raise ValueError("val twins found: R3 makes train + test twins only; check the files")

    nli = pd.concat([pd.read_parquet(p) for p in args.nli], ignore_index=True)
    relevance = None if args.no_relevance else \
        pd.concat([pd.read_parquet(p) for p in args.relevance], ignore_index=True)
    entities = load_entities(args.entities)
    feats = lambda recs: build_features(recs, nli, relevance, entities, args.fill_relevance)
    # Training table: train originals + (kept) train twins; twins only matter if λ > 0.
    use_twins = args.lambda_inv > 0 or args.lambda_sens > 0
    tr = feats(train_recs + (train_tw if use_twins else []))
    va, va_tw = feats(val_recs), feats(val_tw) if val_tw else None
    te = feats(test_recs + test_tw)
    # The held-out twins' originals (they stay in training, their predictions are
    # what consistency is measured against).
    val_tw_bases = {split_twin_id(t["id"])[0] for t in val_tw}
    val_tw_origs = [r for r in train_recs if r["id"] in val_tw_bases]
    tr_orig_for_val_tw = feats(val_tw_origs) if val_tw else None

    features = [f for f in FEATURES if not (args.drop_entity and f in ENTITY_FEATURES)]
    tr_orig = tr[tr["twin_type"].isna()]
    mean, sd = fit_scaler(tr_orig, features)
    config = {"features": features, "mean": mean, "sd": sd, "name": args.name,
              "seed": args.seed, "lambda_inv": args.lambda_inv, "lambda_sens": args.lambda_sens,
              "no_relevance": args.no_relevance, "drop_entity": args.drop_entity,
              "nli_files": args.nli, "lr": args.lr, "batch_groups": args.batch_groups}

    model = VerdictHead(len(features))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = class_weights(list(tr_orig["label"]))
    x_tr = standardise(tr, features, mean, sd)
    groups = make_groups(tr)
    rng = random.Random(args.seed)
    run = start_wandb(args, config)

    best = {"f1": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        logs = defaultdict(float)
        n_batches = 0
        for rows in batches(groups, args.batch_groups, rng):
            kinds, targets, orig_pos = batch_tensors(tr, rows)
            loss, parts = twin_loss(model(x_tr[rows]), kinds, targets, orig_pos, weights,
                                    args.lambda_inv, args.lambda_sens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in {"loss": loss.item(), **parts}.items():
                logs[k] += v
            n_batches += 1
        val_pred = [LABELS[i] for i in predict(model, config, va).argmax(axis=1)]
        val_f1 = macro_f1(list(va["label"]), val_pred) or 0.0
        epoch_log = {k: v / max(n_batches, 1) for k, v in logs.items()}
        epoch_log.update({"epoch": epoch, "val_macro_f1": val_f1})
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in epoch_log.items()))
        if run:
            run.log(epoch_log)
        if val_f1 > best["f1"]:
            best = {"f1": val_f1, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs")
            break
    model.load_state_dict(best["state"])
    config.update({"best_epoch": best["epoch"], "val_macro_f1": best["f1"]})

    tag = f"{args.name}_s{args.seed}"
    out_dir = Path(args.results_dir)
    save_head(model, config, Path(args.models_dir) / f"verdict_{tag}")

    test_preds = pred_rows(te, predict(model, config, te))
    write_jsonl(test_preds, out_dir / f"preds_{tag}.jsonl")
    val_logits = predict(model, config, va)
    write_jsonl([{"id": i, "gold": g, "logits": [round(float(v), 6) for v in row]}
                 for i, g, row in zip(va["id"], va["label"], val_logits)],
                out_dir / f"val_logits_{tag}.jsonl")

    metrics = {"test": evaluate(test_preds, test_recs + test_tw),
               "val": evaluate(pred_rows(va, val_logits), val_recs)}
    if val_tw:  # validation twin metrics: held-out twins + their train originals
        both = pd.concat([tr_orig_for_val_tw, va_tw], ignore_index=True)
        metrics["val_twins"] = evaluate(pred_rows(both, predict(model, config, both)),
                                        val_tw_origs + val_tw)
    metrics["config"] = {k: v for k, v in config.items() if k not in ("mean", "sd")}
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in metrics if k != "config"}, indent=2))
    if run:
        run.summary.update({f"test/{k}": v for k, v in metrics["test"].items() if v is not None})
        run.finish()
    return metrics


def start_wandb(args: argparse.Namespace, config: dict):
    """Start a W&B run in project ``twincheck`` (None if disabled or wandb is missing)."""
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed: not logging (install it and log the real runs)")
        return None
    return wandb.init(project="twincheck", name=f"H3_{args.name}_s{args.seed}",
                      config={k: v for k, v in config.items() if k not in ("mean", "sd")})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options (defaults follow CLAUDE.md §3)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="v0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", nargs="+", default=[f"{EN}/multifc_train.jsonl"])
    ap.add_argument("--val", nargs="+", default=[f"{EN}/multifc_val.jsonl"])
    ap.add_argument("--test", nargs="+", default=[f"{EN}/multifc_test.jsonl"])
    ap.add_argument("--twins", nargs="*", default=[])
    ap.add_argument("--nli", nargs="+", required=True)
    ap.add_argument("--relevance", nargs="+", default=["features/relevance_en.parquet"])
    ap.add_argument("--entities", default="features/entities_en.jsonl")
    ap.add_argument("--fill-relevance", type=float, default=1.0,
                    help="relevance for snippets with no score")
    ap.add_argument("--lambda-inv", type=float, default=0.0)
    ap.add_argument("--lambda-sens", type=float, default=0.0)
    ap.add_argument("--val-twin-frac", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-groups", type=int, default=64)
    ap.add_argument("--no-relevance", action="store_true", help="ablation: no reranker")
    ap.add_argument("--drop-entity", action="store_true", help="ablation: no entity match")
    ap.add_argument("--limit", type=int, help="first N records per split (quick check)")
    ap.add_argument("--results-dir", default="results/H3")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--no-wandb", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
