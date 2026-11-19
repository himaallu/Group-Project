"""Zero-shot NLI baseline: verdict from untrained NLI scores with a simple rule.

What it does
    Uses the per-snippet scores of the base (not fine-tuned) mDeBERTa-v3 XNLI model,
    already computed by Himasri (H3, ``src/nli/score_zeroshot.py``) in
    ``features/nli_zeroshot/*.parquet`` (``id, snippet_idx, p_entail, p_contradict,
    p_neutral``). No training; one rule with one threshold:

        s = max p_entail over snippets, c = max p_contradict over snippets
        if max(s, c) < t        -> unverified
        elif s >= c             -> supported
        else                    -> contradicted

    The threshold t is chosen on the validation scores (macro-F1, grid 0.05 ... 0.95).
    Probabilities: (s, c, 1 - max(s, c)) normalised to sum to 1. A record with no
    scored snippet is ``unverified``. Deterministic: no seeds.

Inputs
    --val-records / --val-nli: labelled records + their NLI parquet (to pick t).
    --predict / --predict-nli: records (originals and/or twins) + NLI parquet(s).
    Works for any language that has NLI scores (e.g. X-FACT AR, Arabic twins).

Outputs
    --out: ``results/<task>/preds_zeroshot_nli.jsonl`` (CLAUDE.md §5).

How to run
    python -m src.eval.baselines.zeroshot_nli \
        --val-records data/processed/en/multifc_val.jsonl --val-nli features/nli_zeroshot/multifc_val.parquet \
        --predict data/processed/en/multifc_test.jsonl data/copies/en/test.jsonl \
        --predict-nli features/nli_zeroshot/multifc_test.parquet features/nli_zeroshot/copies_en/test.parquet \
        --out results/F4/preds_zeroshot_nli.jsonl
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.eval.baselines.common import label_ids, load_records, to_preds, write_preds

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def max_scores(nli_paths: list[str]) -> pd.DataFrame:
    """Per record id: max p_entail and max p_contradict over its snippets."""
    nli = pd.concat([pd.read_parquet(p) for p in nli_paths], ignore_index=True)
    return nli.groupby("id")[["p_entail", "p_contradict"]].max()


def rule_proba(records: list[dict], scores: pd.DataFrame, t: float) -> np.ndarray:
    """(n, 3) probabilities in LABELS order (supported, contradicted, unverified)."""
    s = scores["p_entail"].reindex([r["id"] for r in records]).fillna(0).to_numpy()
    c = scores["p_contradict"].reindex([r["id"] for r in records]).fillna(0).to_numpy()
    top = np.maximum(s, c)
    proba = np.stack([s, c, 1 - top], axis=1) + 1e-9
    # Below the threshold the rule abstains: make "unverified" the arg-max.
    low = top < t
    proba[low, 2] = proba[low, :2].max(axis=1) + 1e-6
    return proba


def pick_threshold(records: list[dict], scores: pd.DataFrame) -> tuple[float, float]:
    """Threshold with the best validation macro-F1; returns (t, macro-F1)."""
    y = label_ids(records)
    best = max(THRESHOLDS, key=lambda t: f1_score(
        y, rule_proba(records, scores, t).argmax(1), average="macro"))
    return float(best), f1_score(y, rule_proba(records, scores, best).argmax(1), average="macro")


def main() -> None:
    """Pick t on validation, predict every --predict file."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--val-records", nargs="+", required=True)
    ap.add_argument("--val-nli", nargs="+", required=True)
    ap.add_argument("--predict", nargs="+", required=True)
    ap.add_argument("--predict-nli", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    val = load_records(args.val_records)
    t, val_f1 = pick_threshold(val, max_scores(args.val_nli))
    records = load_records(args.predict)
    scores = max_scores(args.predict_nli)
    missing = sum(r["id"] not in scores.index for r in records)
    write_preds(to_preds(records, rule_proba(records, scores, t)), args.out)
    print(f"zero-shot NLI: t={t}, val macro-F1 {val_f1:.3f}, {len(records)} predictions "
          f"({missing} without NLI scores -> unverified) -> {args.out}")


if __name__ == "__main__":
    main()
