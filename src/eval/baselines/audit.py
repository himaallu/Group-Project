"""Audit table: every metric for every baseline, per dataset, mean +- sd over seeds.

What it does
    Joins each predictions file with the gold records (``src/eval/loaders.join_gold``),
    splits the rows by dataset (the ``source`` prefix of the base id: multifc, averitec,
    xfact, m4fc, verite) and runs ``src/eval/metrics.compute_all`` on each part. Files
    named ``preds_<model>_s<seed>.jsonl`` are averaged over seeds (mean and sd).

Inputs
    --preds: predictions files (CLAUDE.md §5). --gold: shared-format files (originals
    and twins) that hold the labels.

Outputs
    --out: CSV with one row per (model, dataset): n_seeds, n_orig, n_<twin>, and
    ``<metric>`` / ``<metric>_sd`` for macro_f1, auroc, ece, esr, unrelated_rate,
    paraphrase_consistency, crosslingual_consistency, alr.

How to run
    python -m src.eval.baselines.audit --preds results/F4/preds_*.jsonl \
        --gold data/processed/en/multifc_test.jsonl data/copies/en/test.jsonl \
        --out results/F4/audit_E0.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.baselines.common import load_records
from src.eval.loaders import join_gold, load_preds
from src.eval.metrics import compute_all
from src.eval.report import parse_pred_path

METRICS = ("macro_f1", "auroc", "ece", "esr", "unrelated_rate", "paraphrase_consistency",
           "crosslingual_consistency", "alr")


def evaluate(pred_path: Path, gold: list[dict]) -> list[dict]:
    """Metrics of one predictions file, one row per dataset found in it."""
    _, model, seed = parse_pred_path(pred_path)
    df, _ = join_gold(load_preds(pred_path), gold)
    df["dataset"] = df["base_id"].str.split("_").str[0]
    rows = []
    for dataset, part in df.groupby("dataset"):
        rows.append({"model": model, "seed": seed, "dataset": dataset, **compute_all(part)})
    return rows


def aggregate(per_file: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd over seeds for each (model, dataset)."""
    out = []
    for (model, dataset), part in per_file.groupby(["model", "dataset"], sort=False):
        row = {"model": model, "dataset": dataset, "n_seeds": len(part)}
        for col in [c for c in part.columns if c.startswith("n_")]:
            row[col] = int(part[col].iloc[0])
        for metric in METRICS:
            values = part[metric].dropna()
            row[metric] = values.mean() if len(values) else None
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else None
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    """Evaluate all --preds files against --gold and write the audit CSV."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--gold", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gold = load_records(args.gold)
    per_file = pd.DataFrame([row for p in sorted(args.preds) for row in evaluate(Path(p), gold)])
    table = aggregate(per_file)
    table.to_csv(args.out, index=False)
    show = ["model", "dataset", "n_seeds", "n_orig", "macro_f1", "auroc", "esr",
            "unrelated_rate", "paraphrase_consistency"]
    print(table[[c for c in show if c in table]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
