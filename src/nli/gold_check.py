"""Score any per-snippet NLI feature files against the hand-labelled gold pairs.

What it does
    The gold pairs (``data/nli/gold_val.jsonl``, 150 EN from MultiFC val + 150 AR from
    X-FACT Arabic) are real (claim, snippet) pairs, so their NLI scores are already
    inside the feature files (``id`` = claim id, ``snippet_idx``). This looks them up,
    takes the argmax label and reports macro-F1, accuracy and how often each label is
    predicted, per language. Used to decide which NLI features the verdict head uses:
    fine-tuned scores only replace zero-shot ones if they score higher here.

Inputs
    --features  one or more folders (or parquet files) with columns
                id, snippet_idx, p_entail, p_contradict, p_neutral
    --gold      data/nli/gold_val.jsonl

Outputs
    Printed table; optional JSON with --out.

How to run
    python -m src.nli.gold_check --features features/nli_zeroshot features/nli_v1 features/nli_v2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

NLI_LABELS = ("entail", "contradict", "neutral")
PROB_COLS = ("p_entail", "p_contradict", "p_neutral")


def load_scores(source: str | Path) -> dict[tuple[str, int], tuple[float, float, float]]:
    """{(id, snippet_idx): (p_entail, p_contradict, p_neutral)} from a folder or file."""
    source = Path(source)
    files = sorted(source.rglob("*.parquet")) if source.is_dir() else [source]
    scores = {}
    for f in files:
        df = pd.read_parquet(f)
        if not set(PROB_COLS) <= set(df.columns):
            continue
        for rid, idx, *p in zip(df["id"], df["snippet_idx"], *(df[c] for c in PROB_COLS)):
            scores[(rid, int(idx))] = tuple(float(v) for v in p)
    return scores


def evaluate(scores: dict, gold: list[dict]) -> dict:
    """Macro-F1 / accuracy / predicted-label counts per language (only pairs found).

    Returns:
        {language: {"n", "n_missing", "macro_f1", "accuracy", "pred_counts", "gold_counts"}}
    """
    out = {}
    for lang in sorted({g["language"] for g in gold}):
        rows = [g for g in gold if g["language"] == lang]
        found = [(g["label"], scores[(g["claim_id"], g["snippet_idx"])]) for g in rows
                 if (g["claim_id"], g["snippet_idx"]) in scores]
        if not found:
            out[lang] = {"n": 0, "n_missing": len(rows)}
            continue
        y_true = [t for t, _ in found]
        y_pred = [NLI_LABELS[max(range(3), key=lambda i: p[i])] for _, p in found]
        out[lang] = {
            "n": len(found), "n_missing": len(rows) - len(found),
            "macro_f1": f1_score(y_true, y_pred, labels=list(NLI_LABELS), average="macro",
                                 zero_division=0),
            "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(found),
            "pred_counts": dict(Counter(y_pred)), "gold_counts": dict(Counter(y_true)),
        }
    return out


def main() -> None:
    """CLI: one line per feature source and language."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--gold", default="data/nli/gold_val.jsonl")
    ap.add_argument("--out", help="optional JSON output")
    args = ap.parse_args()
    gold = [json.loads(line) for line in open(args.gold, encoding="utf-8") if line.strip()]
    result = {}
    for src in args.features:
        result[src] = evaluate(load_scores(src), gold)
        for lang, r in result[src].items():
            if r["n"] == 0:
                print(f"{src:32} {lang:3} no gold pairs found ({r['n_missing']} missing)")
                continue
            print(f"{src:32} {lang:3} macro-F1 {r['macro_f1']:.3f}  acc {r['accuracy']:.3f}  "
                  f"n={r['n']} (missing {r['n_missing']})  predicted {r['pred_counts']}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
