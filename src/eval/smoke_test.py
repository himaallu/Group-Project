"""Smoke-test the evaluation code on a real hand-off file (e.g. a 10-row sample).

What it does
    1. Checks every record against the shared format (``src.eval.loaders``).
    2. Makes random predictions for every record (seed 42), writes them in the
       predictions format, reads them back and joins them with the gold labels.
    3. Runs ``compute_all`` and prints the metrics. The numbers mean nothing; the
       point is that every step runs and the counts look right.

Inputs
    One or more shared-format ``.jsonl`` files (originals and/or twins).

Outputs
    ``<out>/preds_random.jsonl`` (default ``results/H1/smoke/``) and printed metrics.

How to run
    python -m src.eval.smoke_test data/processed/en/multifc_sample10.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.loaders import LABELS, join_gold, load_jsonl, load_preds, summarise_file
from src.eval.metrics import compute_all


def random_preds(records: list[dict], seed: int = 42) -> list[dict]:
    """Make one random prediction per record in the predictions format.

    Args:
        records: shared-format records.
        seed: random seed.

    Returns:
        List of ``{"id", "pred_label", "probs"}`` dicts (probs from a Dirichlet).
    """
    rng = np.random.default_rng(seed)
    preds = []
    for rec in records:
        p = rng.dirichlet(np.ones(len(LABELS)))
        probs = {lab: float(v) for lab, v in zip(LABELS, p)}
        preds.append({"id": rec["id"], "pred_label": LABELS[int(p.argmax())], "probs": probs})
    return preds


def run(paths: list[Path], out_dir: Path) -> dict:
    """Check the files, write random predictions, and compute all metrics.

    Args:
        paths: shared-format gold files.
        out_dir: folder for ``preds_random.jsonl``.

    Returns:
        The metrics dict from ``compute_all``.
    """
    records = []
    for path in paths:
        summary = summarise_file(path)
        print(f"== {path}: {summary['n_records']} records, {summary['n_invalid']} invalid")
        for key in ("label", "split", "language", "source", "twin_type",
                    "snippets_min_mean_max", "n_no_snippets", "duplicate_ids"):
            print(f"   {key}: {summary[key]}")
        for problem in summary["problems"]:
            print(f"   PROBLEM {problem}")
        records += load_jsonl(path)

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "preds_random.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in random_preds(records):
            f.write(json.dumps(p) + "\n")

    # Read back through the same path the real models use.
    df, missing = join_gold(load_preds(pred_path), records)
    print(f"joined {len(df)} rows ({len(missing)} without gold)")
    metrics = compute_all(df)
    for key, value in metrics.items():
        print(f"   {key}: {value if value is None or isinstance(value, int) else round(value, 4)}")
    return metrics


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Smoke-test metrics on a hand-off file.")
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("results/H1/smoke"))
    args = parser.parse_args()
    run(args.files, args.out)


if __name__ == "__main__":
    main()
