"""H5 experiments that need no teammate hand-off: ablations, AVeriTeC, calibration, bootstrap.

What it does (CLAUDE.md §9 H5, §5)
    1. Trains four variants of the verdict head with seeds 42/43/44 on MultiFC train
       + train twins, predicting MultiFC test + test twins (E2 and ablations):
           twin        chosen λ from H4 (results/H4/selected_lambda.json)
           notwin      λ_inv = λ_sens = 0            (ablation: no twin loss)
           twin_norel  chosen λ, no reranker          (ablation: no reranker)
           twin_noent  chosen λ, no entity match      (ablation: no entity match)
       The "zero-shot NLI" ablation is the comparison between a run with
       ``--nli-dir features/nli_zeroshot`` and one with Rathi's fine-tuned features.
    2. Applies every model to AVeriTeC test (E1, second English test set): avt_<variant>.
    3. Calibration (E5) for twin and notwin: T and τ fitted on MultiFC validation,
       applied to MultiFC test and AVeriTeC test: <variant>_cal, avt_<variant>_cal.
    4. Paired bootstrap (1,000 resamples): twin vs each ablation on MultiFC; twin vs
       notwin on AVeriTeC; per seed.
    5. Mean ± sd table over seeds for every model.

    ``--tag`` is appended to every model name, so a rerun with fine-tuned NLI
    (``--nli-dir features/nli_v1 --tag _r4``) sits next to the zero-shot run.

Inputs
    --nli-dir with multifc_{train,val,test}.parquet, twin scores copies_en/{train,test}.parquet
              (or {train,test}.parquet),
              averitec_test.parquet (same columns as score_zeroshot.py writes)
    data/processed/en/{multifc_*,averitec_test}.jsonl, data/copies/en/{train,test}.jsonl,
    features/relevance_en.parquet, features/relevance_copies_en.parquet,
    features/entities_en.jsonl

Outputs
    results/H5/preds_<model>_s<seed>.jsonl (+ val_logits_*, metrics_* for trained ones)
    results/H5/bootstrap<tag>.json, results/H5/summary<tag>.csv
    models/H5/verdict_<variant><tag>_s<seed>/, models/H5/verdict_calib_<variant><tag>_s<seed>.json

How to run (Colab; training on CPU, ~25-40 min in total)
    python -m src.eval.run_h5 --no-wandb
    python -m src.eval.run_h5 --nli-dir features/nli_v1 --tag _r4 --no-wandb   # with R4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.eval.compare import compare
from src.eval.loaders import join_gold, load_jsonl, load_preds
from src.eval.metrics import compute_all
from src.verdict import calibrate as C
from src.verdict import train as T
from src.verdict.features import load_entities
from src.verdict.predict import predict_records

EN = "data/processed/en"
TWINS = ["data/copies/en/train.jsonl", "data/copies/en/test.jsonl"]
RELEVANCE = ["features/relevance_en.parquet", "features/relevance_copies_en.parquet"]
ENTITIES = "features/entities_en.jsonl"
AVERITEC = f"{EN}/averitec_test.jsonl"
CALIBRATED = ("twin", "notwin")
SUMMARY_COLS = ("macro_f1", "auroc", "ece", "esr", "unrelated_rate", "paraphrase_consistency")


def variants(lambda_inv: float, lambda_sens: float) -> dict[str, list[str]]:
    """Variant name -> extra train.py arguments."""
    lam = ["--lambda-inv", str(lambda_inv), "--lambda-sens", str(lambda_sens)]
    return {"twin": lam,
            "notwin": ["--lambda-inv", "0", "--lambda-sens", "0"],
            "twin_norel": lam + ["--no-relevance"],
            "twin_noent": lam + ["--drop-entity"]}


def nli_files(nli_dir: Path) -> dict[str, list[str]]:
    """NLI parquet paths for training (MultiFC + twins) and for AVeriTeC test."""
    train = [nli_dir / f"multifc_{s}.parquet" for s in ("train", "val", "test")]
    # Twin scores: copies_en/{train,test}.parquet (score_zeroshot with --out-dir), or
    # {train,test}.parquet next to the others (Rathi's src/nli/score.py names by file stem).
    for s in ("train", "test"):
        nested = nli_dir / "copies_en" / f"{s}.parquet"
        train.append(nested if nested.exists() else nli_dir / f"{s}.parquet")
    files = {"train": [str(p) for p in train], "averitec": [str(nli_dir / "averitec_test.parquet")]}
    missing = [p for group in files.values() for p in group if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"missing NLI files: {missing}")
    return files


def summarise(out_dir: Path, gold: list[dict], tag: str) -> pd.DataFrame:
    """Mean ± sd over seeds for every ``preds_*<tag>_s*.jsonl`` in ``out_dir``."""
    rows = []
    pattern = re.compile(rf"preds_(.+{re.escape(tag)}(?:_cal)?)_s(\d+)\.jsonl$")
    for path in sorted(out_dir.glob("preds_*_s*.jsonl")):
        m = pattern.search(path.name)
        if not m:
            continue
        df, _ = join_gold(load_preds(path), gold)
        rows.append({"model": m.group(1), "seed": int(m.group(2)), **compute_all(df)})
    res = pd.DataFrame(rows)
    table = res.groupby("model")[list(SUMMARY_COLS)].agg(["mean", "std"])
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    return table.round(4)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Run all H5 steps (see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nli-dir", default="features/nli_zeroshot")
    ap.add_argument("--tag", default="", help="suffix for model names, e.g. _r4")
    ap.add_argument("--out-dir", default="results/H5")
    ap.add_argument("--models-dir", default="models/H5", help="kept apart from the H4 models")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--selected", default="results/H4/selected_lambda.json")
    ap.add_argument("--only", nargs="+", help="train only these variants (default: all)")
    ap.add_argument("--en-dir", default=EN)
    ap.add_argument("--twins", nargs="+", default=TWINS)
    ap.add_argument("--relevance", nargs="+", default=RELEVANCE)
    ap.add_argument("--entities", default=ENTITIES)
    ap.add_argument("--averitec", default=AVERITEC)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    out, tag = Path(args.out_dir), args.tag
    chosen = json.loads(Path(args.selected).read_text())["chosen"]
    nli = nli_files(Path(args.nli_dir))
    todo = {k: v for k, v in variants(chosen["lambda_inv"], chosen["lambda_sens"]).items()
            if not args.only or k in args.only}
    base_argv = ["--train", f"{args.en_dir}/multifc_train.jsonl",
                 "--val", f"{args.en_dir}/multifc_val.jsonl",
                 "--test", f"{args.en_dir}/multifc_test.jsonl",
                 "--twins", *args.twins, "--nli", *nli["train"],
                 "--relevance", *args.relevance, "--entities", args.entities,
                 "--epochs", str(args.epochs),
                 "--results-dir", str(out), "--models-dir", args.models_dir]
    if args.no_wandb:
        base_argv.append("--no-wandb")

    # 1. train
    for name, extra in todo.items():
        for seed in args.seeds:
            print(f"\n=== train {name}{tag} seed {seed} ===", flush=True)
            T.train(T.parse_args(base_argv + extra + ["--name", f"{name}{tag}", "--seed", str(seed)]))

    # 2. AVeriTeC predictions
    avt_records = load_jsonl(args.averitec)
    avt_nli = pd.concat([pd.read_parquet(p) for p in nli["averitec"]], ignore_index=True)
    relevance = pd.concat([pd.read_parquet(p) for p in args.relevance], ignore_index=True)
    entities = load_entities(args.entities)
    for name in todo:
        for seed in args.seeds:
            model_dir = Path(args.models_dir) / f"verdict_{name}{tag}_s{seed}"
            rows = predict_records(model_dir, avt_records, avt_nli, relevance, entities)
            T.write_jsonl(rows, out / f"preds_avt_{name}{tag}_s{seed}.jsonl")
    print("AVeriTeC predictions written", flush=True)

    # 3. calibration fitted on MultiFC val, applied to MultiFC test and AVeriTeC test
    for name in [n for n in CALIBRATED if n in todo]:
        for seed in args.seeds:
            m = f"{name}{tag}"
            _, logits, gold_val = C.load_logits(out / f"val_logits_{m}_s{seed}.jsonl")
            settings = C.fit(logits, gold_val)
            C.save_settings(settings, Path(args.models_dir) / f"verdict_calib_{m}_s{seed}.json")
            C.calibrate_pred_file(settings, out / f"preds_{m}_s{seed}.jsonl",
                                  out / f"preds_{m}_cal_s{seed}.jsonl")
            C.calibrate_pred_file(settings, out / f"preds_avt_{m}_s{seed}.jsonl",
                                  out / f"preds_avt_{m}_cal_s{seed}.jsonl")
    print("calibration done", flush=True)

    # 4. paired bootstrap per seed
    gold = (load_jsonl(f"{args.en_dir}/multifc_test.jsonl")
            + [r for p in args.twins for r in load_jsonl(p)] + avt_records)
    pairs = [(f"twin{tag}", f"{other}{tag}") for other in ("notwin", "twin_norel", "twin_noent")]
    pairs += [(f"avt_twin{tag}", f"avt_notwin{tag}")]
    boot = {}
    for a, b in pairs:
        for seed in args.seeds:
            pa, pb = out / f"preds_{a}_s{seed}.jsonl", out / f"preds_{b}_s{seed}.jsonl"
            if pa.exists() and pb.exists():
                boot[f"{a} vs {b} (s{seed})"] = compare(pa, pb, gold)
    (out / f"bootstrap{tag}.json").write_text(json.dumps(boot, indent=2) + "\n")

    # 5. summary
    table = summarise(out, gold, tag)
    table.to_csv(out / f"summary{tag}.csv")
    pd.set_option("display.width", 250)
    print(table.to_string())
    return table


if __name__ == "__main__":
    main()
