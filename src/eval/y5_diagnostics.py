"""Extra E7 checks on the Arabic twin set (task Y5/Y6).

The E7 table alone can hide a class bias: a head that calls almost everything
"contradicted" gets a high ESR for free. This script adds, per verdict head and
language, mean over seeds 42/43/44:
    supported_recall   % of supported twins (English original or _A) predicted supported
    paired_esr         % of swap twins (_B / _B_A) predicted contradicted, counting only
                       sources whose pre-swap twin was NOT predicted contradicted
    paired_n           number of such sources
    errors_inherited   % of wrong Arabic/mixed verdicts whose English twin is also wrong

Outputs results/Y5/diagnostics.csv.
How to run (CPU, seconds): python -m src.eval.y5_diagnostics
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.loaders import load_jsonl, split_twin_id
from src.eval.run_y5_ar import SEEDS, VARIANTS, kind

PAIRS = {"en": ("orig", "B"), "ar": ("A", "B_A")}
EN_OF = {"A": "orig", "M": "orig", "B_A": "B", "C_A": "C"}


def by_source(preds: list[dict]) -> dict[str, dict[str, str]]:
    """{base id: {twin kind: predicted label}} for the twin set."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for p in preds:
        if not p["id"].startswith("xfact"):
            out[split_twin_id(p["id"])[0]][kind(p["id"])] = p["pred_label"]
    return out


def diagnostics(src: dict[str, dict[str, str]], gold: dict[str, dict[str, str]]) -> list[dict]:
    """Rows for one head and seed."""
    rows = []
    for lang, (pre, post) in PAIRS.items():
        have = [d for d in src.values() if pre in d]
        eligible = [d for d in have if post in d and d[pre] != "contradicted"]
        rows.append({"language": lang,
                     "supported_recall": np.mean([d[pre] == "supported" for d in have]),
                     "paired_esr": np.mean([d[post] == "contradicted" for d in eligible]) if eligible else np.nan,
                     "paired_n": len(eligible)})
    wrong = [(b, k) for b, d in src.items() for k in EN_OF if k in d and d[k] != gold[b][k]]
    inherited = [src[b].get(EN_OF[k]) != gold[b].get(EN_OF[k]) for b, k in wrong if EN_OF[k] in src[b]]
    rows[1]["errors_inherited"] = np.mean(inherited) if inherited else np.nan
    rows[1]["errors_n"] = len(wrong)
    return rows


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds-dir", default="results/Y5")
    ap.add_argument("--gold", nargs="+", default=["data/copies/ar/test.jsonl", "data/copies/en/test.jsonl",
                                                  "data/processed/en/multifc_test.jsonl"])
    args = ap.parse_args(argv)
    gold: dict[str, dict[str, str]] = defaultdict(dict)
    for path in args.gold:
        for r in load_jsonl(path):
            gold[split_twin_id(r["id"])[0]][kind(r["id"])] = r["label"]
    rows = []
    for v in VARIANTS:
        for s in SEEDS:
            path = Path(args.preds_dir) / f"preds_{v}_s{s}.jsonl"
            if path.exists():
                rows += [{**r, "system": v, "seed": s} for r in diagnostics(by_source(load_jsonl(path)), gold)]
    df = pd.DataFrame(rows)
    table = df.groupby(["system", "language"], sort=False).mean(numeric_only=True).drop(columns="seed").round(3)
    table.to_csv(Path(args.preds_dir) / "diagnostics.csv")
    print(table)
    return table


if __name__ == "__main__":
    main()
