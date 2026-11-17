"""Apply a trained verdict head to any data file (H5: AVeriTeC; Y5: Arabic test sets).

What it does
    Loads ``models/verdict_<name>/`` (weights + config), builds the same features
    the model was trained with (``features.py``; respects its ``no_relevance`` and
    feature list) and writes §5 predictions.

Inputs
    --model      model folder, e.g. models/verdict_twin_s42
    --data       shared-format records (originals and/or twins)
    --nli        per-snippet NLI parquet file(s) covering every record
    --relevance  per-snippet relevance parquet file(s) (ignored for --no-relevance models)
    --entities   R2 entities file

Outputs
    --out        predictions JSONL ``{"id", "pred_label", "probs"}``

How to run
    python -m src.verdict.predict --model models/verdict_twin_s42 \
        --data data/processed/en/averitec_test.jsonl \
        --nli features/nli_zeroshot/averitec_test.parquet \
        --out results/H5/preds_avt_twin_s42.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.loaders import load_jsonl
from src.verdict.features import build_features, load_entities
from src.verdict.model import load_head, predict
from src.verdict.train import pred_rows, write_jsonl


def predict_records(model_dir: str | Path, records: list[dict], nli: pd.DataFrame,
                    relevance: pd.DataFrame | None, entities: dict,
                    fill_relevance: float = 1.0) -> list[dict]:
    """§5 prediction rows for ``records`` from a saved verdict head.

    Args:
        model_dir: folder written by ``model.save_head``.
        records: shared-format records.
        nli: per-snippet NLI scores.
        relevance: per-snippet relevance (unused if the model was trained without it).
        entities: R2 entity rows by id.
        fill_relevance: relevance for snippets with no score.

    Returns:
        List of ``{"id", "pred_label", "probs"}`` dicts, in record order.
    """
    model, config = load_head(model_dir)
    rel = None if config.get("no_relevance") else relevance
    table = build_features(records, nli, rel, entities, fill_relevance)
    return pred_rows(table, predict(model, config, table))


def main() -> None:
    """CLI: one model, one or more data files, one predictions file."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--nli", nargs="+", required=True)
    ap.add_argument("--relevance", nargs="+", default=["features/relevance_en.parquet"])
    ap.add_argument("--entities", default="features/entities_en.jsonl")
    ap.add_argument("--fill-relevance", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    records = [r for p in args.data for r in load_jsonl(p)]
    nli = pd.concat([pd.read_parquet(p) for p in args.nli], ignore_index=True)
    relevance = pd.concat([pd.read_parquet(p) for p in args.relevance], ignore_index=True)
    rows = predict_records(args.model, records, nli, relevance,
                           load_entities(args.entities), args.fill_relevance)
    write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
