"""λ grid for the twin loss, choice of the best setting, and final seeds (H4, CLAUDE.md §3).

What it does
    1. Trains the verdict head (``train.py``) with seed 42 for every
       λ_inv ∈ {0, 0.5, 1, 2} × λ_sens ∈ {0, 0.5, 1} (12 runs) on train originals +
       train twins. Outputs go to ``results/H4/grid/`` (not read by report.py).
    2. Picks the setting with the best validation twin score (mean of paraphrase
       consistency, ESR and unrelated rate on the held-out validation twins) among
       those whose validation macro-F1 is at most 1 point below the λ = (0, 0) run.
       Ties go to the smaller λ_inv + λ_sens.
    3. Reruns the chosen setting as model ``twin`` and λ = (0, 0) as model ``notwin``
       with seeds 42 / 43 / 44 into ``results/H4/`` (E2: with vs without twin loss).

Inputs
    Same data / feature arguments as ``train.py`` (passed through unchanged), which
    must include ``--twins`` (train + test twin files).

Outputs
    results/H4/grid/{preds,val_logits,metrics}_inv<a>_sens<b>_s42.*
    results/H4/grid_summary.csv          one row per grid run
    results/H4/selected_lambda.json      chosen λ and why
    results/H4/{preds,val_logits,metrics}_{twin,notwin}_s{42,43,44}.*
    models/verdict_{twin,notwin}_s{42,43,44}/

How to run (Colab; the head trains on CPU, ~2-3 min per run)
    python -m src.verdict.grid --twins data/copies/en/train.jsonl data/copies/en/test.jsonl \
        --nli features/nli_zeroshot/multifc_{train,val,test}.parquet \
              features/nli_zeroshot/copies_en/{train,test}.parquet \
        --relevance features/relevance_en.parquet features/relevance_copies_en.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.verdict import train as T

LAMBDA_INV = (0.0, 0.5, 1.0, 2.0)
LAMBDA_SENS = (0.0, 0.5, 1.0)
TWIN_SCORE_KEYS = ("paraphrase_consistency", "esr", "unrelated_rate")
MAX_F1_DROP = 0.01      # "≤ 1 point macro-F1 loss" (CLAUDE.md §3)
FINAL_SEEDS = (42, 43, 44)


def run_name(lambda_inv: float, lambda_sens: float) -> str:
    """Model name for a grid point, e.g. ``inv0.5_sens1``."""
    fmt = lambda x: f"{x:g}"
    return f"inv{fmt(lambda_inv)}_sens{fmt(lambda_sens)}"


def twin_score(val_twins: dict) -> float | None:
    """Mean of the available validation twin metrics (None if none is available)."""
    values = [val_twins.get(k) for k in TWIN_SCORE_KEYS if val_twins.get(k) is not None]
    return sum(values) / len(values) if values else None


def summarise_run(lambda_inv: float, lambda_sens: float, metrics: dict) -> dict:
    """One grid_summary row from a run's metrics dict."""
    vt = metrics.get("val_twins", {})
    return {"name": run_name(lambda_inv, lambda_sens), "lambda_inv": lambda_inv,
            "lambda_sens": lambda_sens, "val_macro_f1": metrics["val"]["macro_f1"],
            **{f"val_{k}": vt.get(k) for k in TWIN_SCORE_KEYS},
            "val_twin_score": twin_score(vt), "best_epoch": metrics["config"]["best_epoch"]}


def select(rows: list[dict], max_f1_drop: float = MAX_F1_DROP) -> dict:
    """Choose the grid point (see module docstring, step 2).

    Args:
        rows: ``summarise_run`` rows; must include λ = (0, 0).
        max_f1_drop: allowed validation macro-F1 loss vs λ = (0, 0).

    Returns:
        {"chosen": row, "base_val_macro_f1": float, "eligible": [names]}.

    Raises:
        ValueError: if the λ = (0, 0) run is missing.
    """
    base = next((r for r in rows if r["lambda_inv"] == 0 and r["lambda_sens"] == 0), None)
    if base is None:
        raise ValueError("the grid must include lambda_inv = lambda_sens = 0")
    floor = base["val_macro_f1"] - max_f1_drop
    eligible = [r for r in rows if r["val_macro_f1"] >= floor and r["val_twin_score"] is not None]
    # best twin score; ties -> smaller total λ; the base run is always eligible
    chosen = min(eligible, key=lambda r: (-round(r["val_twin_score"], 6),
                                          r["lambda_inv"] + r["lambda_sens"]))
    return {"chosen": chosen, "base_val_macro_f1": base["val_macro_f1"],
            "f1_floor": floor, "eligible": [r["name"] for r in eligible]}


def train_one(passthrough: list[str], name: str, seed: int, lambda_inv: float,
              lambda_sens: float, results_dir: str, models_dir: str) -> dict:
    """Run ``train.train`` with the shared arguments plus this run's settings."""
    argv = passthrough + ["--name", name, "--seed", str(seed),
                          "--lambda-inv", str(lambda_inv), "--lambda-sens", str(lambda_sens),
                          "--results-dir", results_dir, "--models-dir", models_dir]
    return T.train(T.parse_args(argv))


def main(argv: list[str] | None = None) -> dict:
    """CLI: grid with seed 42 -> select -> final seeds. Unknown options go to train.py."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="results/H4")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--lambda-inv", type=float, nargs="+", default=list(LAMBDA_INV))
    ap.add_argument("--lambda-sens", type=float, nargs="+", default=list(LAMBDA_SENS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(FINAL_SEEDS))
    ap.add_argument("--skip-final", action="store_true", help="grid + selection only")
    args, passthrough = ap.parse_known_args(argv)
    if "--twins" not in passthrough:
        ap.error("--twins is required (train + test twin files)")
    out = Path(args.out_dir)
    grid_dir = out / "grid"

    rows = []
    for li in args.lambda_inv:
        for ls in args.lambda_sens:
            print(f"\n=== grid {run_name(li, ls)} (seed 42) ===")
            m = train_one(passthrough, run_name(li, ls), 42, li, ls, str(grid_dir),
                          str(Path(args.models_dir) / "grid"))
            rows.append(summarise_run(li, ls, m))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "grid_summary.csv", index=False)
    choice = select(rows)
    (out / "selected_lambda.json").write_text(json.dumps(choice, indent=2) + "\n")
    c = choice["chosen"]
    print(f"\nchosen: {c['name']} (val twin score {c['val_twin_score']:.4f}, "
          f"val macro-F1 {c['val_macro_f1']:.4f}; floor {choice['f1_floor']:.4f})")

    if not args.skip_final:
        for seed in args.seeds:
            for name, li, ls in (("twin", c["lambda_inv"], c["lambda_sens"]), ("notwin", 0.0, 0.0)):
                print(f"\n=== final {name} seed {seed} ===")
                train_one(passthrough, name, seed, li, ls, str(out), args.models_dir)
    return choice


if __name__ == "__main__":
    main()
