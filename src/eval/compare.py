"""Paired bootstrap comparison of two models' predictions (CLAUDE.md §5: 95% CIs + p-values).

What it does
    Joins two prediction files with the same gold records and, for each metric,
    runs ``metrics.paired_bootstrap`` (1,000 resamples by base id, seed 42) on the
    difference metric(A) - metric(B). With several seeds, pass one file per seed
    for each model in the same order; every seed pair is compared separately.

Inputs
    --a / --b   prediction files (§5 format), e.g. results/H4/preds_twin_s42.jsonl
    --gold      shared-format gold files covering every predicted id

Outputs
    JSON with one entry per seed pair and metric:
        {"a": ..., "b": ..., "diff": ..., "ci_low": ..., "ci_high": ..., "p_value": ...}

How to run
    python -m src.eval.compare --a results/H4/preds_twin_s42.jsonl --b results/H4/preds_notwin_s42.jsonl \
        --gold data/processed/en/multifc_test.jsonl data/copies/en/test.jsonl \
        --out results/H4/bootstrap_twin_vs_notwin.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.loaders import join_gold, load_jsonl, load_preds
from src.eval import metrics as M


def _orig_f1(df):
    """Macro-F1 on the original claims of a joined table."""
    o = M.originals(df)
    return M.macro_f1(list(o["gold"]), list(o["pred"]))


def _orig_ece(df):
    """ECE on the original claims of a joined table."""
    o = M.originals(df)
    return M.ece(o["gold"], o[list(M.PROB_COLS)].to_numpy(), pred=list(o["pred"]))


METRICS = {
    "macro_f1": _orig_f1,
    "ece": _orig_ece,
    "esr": M.evidence_sensitivity,
    "unrelated_rate": M.unrelated_rate,
    "paraphrase_consistency": M.paraphrase_consistency,
    "crosslingual_consistency": M.crosslingual_consistency,
}


def compare(path_a: str | Path, path_b: str | Path, gold: list[dict],
            n_resamples: int = 1000, seed: int = 42) -> dict:
    """Bootstrap every applicable metric for one pair of prediction files.

    Returns:
        {metric: paired_bootstrap result}; metrics with nothing to measure are left out.
    """
    df_a, _ = join_gold(load_preds(path_a), gold)
    df_b, _ = join_gold(load_preds(path_b), gold)
    out = {}
    for name, fn in METRICS.items():
        if fn(df_a) is None or fn(df_b) is None:
            continue
        out[name] = M.paired_bootstrap(df_a, df_b, fn, n_resamples=n_resamples, seed=seed)
    return out


def main() -> None:
    """CLI: compare seed-matched prediction files of two models."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", nargs="+", required=True, help="model A files (one per seed)")
    ap.add_argument("--b", nargs="+", required=True, help="model B files, same seed order")
    ap.add_argument("--gold", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-resamples", type=int, default=1000)
    args = ap.parse_args()
    if len(args.a) != len(args.b):
        ap.error("--a and --b need the same number of files")
    gold = [r for p in args.gold for r in load_jsonl(p)]
    result = {f"{Path(a).stem} vs {Path(b).stem}": compare(a, b, gold, args.n_resamples)
              for a, b in zip(args.a, args.b)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    for pair, res in result.items():
        print(pair)
        for name, r in res.items():
            print(f"  {name:24} A {r['a']:.4f}  B {r['b']:.4f}  diff {r['diff']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  p={r['p_value']:.3f}")


if __name__ == "__main__":
    main()
