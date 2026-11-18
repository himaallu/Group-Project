"""E5 on X-FACT Arabic: does the verdict head's calibration transfer from English? (H5)

What it does
    For each trained verdict head (``twin`` and ``notwin``, seeds 42/43/44, from H5)
    it predicts X-FACT Arabic val and test (``src/verdict/predict.py``), then writes
    three versions of the test predictions:
        xfar_<variant>          no calibration
        xfar_<variant>_calmfc   T and τ fitted on MultiFC val (English), from H5
        xfar_<variant>_calar    T and τ fitted on X-FACT Arabic val (in-language)
    and a mean ± sd table (macro-F1, AUROC, ECE) over seeds.
    Entities: the English spaCy features are not used for Arabic (set to 1.0, as in
    ``src/verdict/infer.py``).

Inputs
    data/processed/ar/xfact_ar_{val,test}.jsonl
    --nli-dir with xfact_ar_val.parquet and xfact_ar_test.parquet
              (``python -m src.nli.score_zeroshot ... --out-dir features/nli_zeroshot``)
    --relevance  features/relevance_ar.parquet (``python -m src.retrieval.rerank``)
    models/H5/verdict_{twin,notwin}_s*/ and models/H5/verdict_calib_{twin,notwin}_s*.json

Outputs
    results/H5/preds_xfar_*_s<seed>.jsonl, results/H5/e5_xfact_ar_summary.csv,
    models/H5/verdict_calib_xfar_{variant}_s<seed>.json (the Arabic-fitted settings)

How to run (CPU is fine once the NLI and relevance files exist)
    python -m src.eval.run_e5_ar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.loaders import join_gold, load_jsonl, load_preds
from src.eval.metrics import LABELS, compute_all
from src.verdict import calibrate as C
from src.verdict.predict import predict_records
from src.verdict.train import write_jsonl

AR = "data/processed/ar"
COLS = ("macro_f1", "auroc", "ece")


def probs_to_logits(rows: list[dict]) -> np.ndarray:
    """log-probabilities as logits (valid for temperature scaling, see calibrate.py)."""
    return np.log(np.array([[r["probs"][lab] for lab in LABELS] for r in rows]) + 1e-12)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Predict, calibrate three ways, summarise (see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--val", default=f"{AR}/xfact_ar_val.jsonl")
    ap.add_argument("--test", default=f"{AR}/xfact_ar_test.jsonl")
    ap.add_argument("--nli-dir", default="features/nli_zeroshot")
    ap.add_argument("--relevance", default="features/relevance_ar.parquet")
    ap.add_argument("--models-dir", default="models/H5")
    ap.add_argument("--variants", nargs="+", default=["twin", "notwin"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out-dir", default="results/H5")
    args = ap.parse_args(argv)

    val, test = load_jsonl(args.val), load_jsonl(args.test)
    nli_dir = Path(args.nli_dir)
    nli = pd.concat([pd.read_parquet(nli_dir / f"{Path(p).stem}.parquet")
                     for p in (args.val, args.test)], ignore_index=True)
    relevance = pd.read_parquet(args.relevance)
    out, models = Path(args.out_dir), Path(args.models_dir)

    for variant in args.variants:
        for seed in args.seeds:
            model_dir = models / f"verdict_{variant}_s{seed}"
            val_rows = predict_records(model_dir, val, nli, relevance, entities={})
            test_rows = predict_records(model_dir, test, nli, relevance, entities={})
            raw = out / f"preds_xfar_{variant}_s{seed}.jsonl"
            write_jsonl(test_rows, raw)
            # English-fitted settings from H5 (transfer)
            mfc = C.load_settings(models / f"verdict_calib_{variant}_s{seed}.json")
            C.calibrate_pred_file(mfc, raw, out / f"preds_xfar_{variant}_calmfc_s{seed}.jsonl")
            # Arabic-fitted settings (in-language)
            ar = C.fit(probs_to_logits(val_rows), [r["label"] for r in val])
            C.save_settings(ar, models / f"verdict_calib_xfar_{variant}_s{seed}.json")
            C.calibrate_pred_file(ar, raw, out / f"preds_xfar_{variant}_calar_s{seed}.jsonl")
            print(f"{variant} s{seed}: English T={mfc['T']:.3f} tau={mfc['tau']} | "
                  f"Arabic T={ar['T']:.3f} tau={ar['tau']}", flush=True)

    rows = []
    for path in sorted(out.glob("preds_xfar_*_s*.jsonl")):
        name, seed = path.stem.removeprefix("preds_").rsplit("_s", 1)
        df, _ = join_gold(load_preds(path), test)
        rows.append({"model": name, "seed": int(seed), **compute_all(df)})
    res = pd.DataFrame(rows)
    table = res.groupby("model")[list(COLS)].agg(["mean", "std"]).round(4)
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table["n_test"] = len(test)
    table.to_csv(out / "e5_xfact_ar_summary.csv")
    print(table.to_string())
    return table


if __name__ == "__main__":
    main()
