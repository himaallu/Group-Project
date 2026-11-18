"""Collect English error cases for the error analysis (H6), grouped by failure type.

What it does
    Joins the twin-loss model's test predictions (and the no-twin model's, for the
    "fixed by the twin loss" group) with the gold records and the per-snippet NLI
    scores, and samples up to ``--per-group`` cases (seed 42) for each group:
        supported_as_contradicted  original, gold supported, predicted contradicted
        contradicted_as_supported  original, gold contradicted, predicted supported
        unverified_missed          original, gold unverified, predicted something else
        missed_swap                _B twin not predicted contradicted
        paraphrase_flip            _P twin whose verdict differs from its original's
        fixed_by_twin_loss         _B twin: no-twin model wrong, twin model right
    Each row shows the claim (and the original claim for twins), the predicted
    probabilities and the top 2 snippets (by relevance) with their NLI scores, plus an
    empty ``cause`` column to fill in by hand (CLAUDE.md H6: 10 cases).

Inputs
    --preds / --base-preds  results/H5/preds_twin_s42.jsonl, results/H5/preds_notwin_s42.jsonl
    --gold                  data/processed/en/multifc_test.jsonl data/copies/en/test.jsonl
    --nli                   features/nli_zeroshot/multifc_test.parquet features/nli_zeroshot/copies_en/test.parquet
    --relevance             features/relevance_en.parquet features/relevance_copies_en.parquet

Outputs
    results/H6/error_cases.csv (UTF-8 with BOM, opens in Excel)

How to run
    python -m src.eval.error_cases
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pandas as pd

from src.eval.loaders import load_jsonl, load_preds, split_twin_id, twin_type

GROUPS = ("supported_as_contradicted", "contradicted_as_supported", "unverified_missed",
          "missed_swap", "paraphrase_flip", "fixed_by_twin_loss")
COLUMNS = ("group", "id", "twin_type", "gold", "pred", "p_supported", "p_contradicted",
           "p_unverified", "base_pred", "claim", "orig_claim", "snippet_1", "nli_1",
           "snippet_2", "nli_2", "cause")


def per_snippet(paths: list[str], cols: list[str]) -> dict[str, dict[int, tuple]]:
    """{id: {snippet_idx: values}} from parquet files."""
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    out: dict[str, dict[int, tuple]] = {}
    for row in df[["id", "snippet_idx", *cols]].itertuples(index=False):
        out.setdefault(row[0], {})[int(row[1])] = tuple(row[2:])
    return out


def group_of(rec_id: str, gold: str, pred: str, orig_pred: str | None,
             base_pred: str | None) -> list[str]:
    """Every failure group a prediction belongs to (may be several or none)."""
    kind = twin_type(rec_id)
    groups = []
    if kind is None:
        if gold == "supported" and pred == "contradicted":
            groups.append("supported_as_contradicted")
        if gold == "contradicted" and pred == "supported":
            groups.append("contradicted_as_supported")
        if gold == "unverified" and pred != "unverified":
            groups.append("unverified_missed")
    elif kind == "B":
        if pred != "contradicted":
            groups.append("missed_swap")
        elif base_pred is not None and base_pred != "contradicted":
            groups.append("fixed_by_twin_loss")
    elif kind == "P" and orig_pred is not None and pred != orig_pred:
        groups.append("paraphrase_flip")
    return groups


def snippet_text(rec: dict, idx: int, nli: dict, max_chars: int = 220) -> tuple[str, str]:
    """(snippet text, "e/c/n" NLI string) for one snippet."""
    text = rec["evidence"][idx]["text"][:max_chars]
    scores = nli.get(rec["id"], {}).get(idx)
    return text, ("/".join(f"{v:.2f}" for v in scores) if scores else "")


def build_cases(preds: list[dict], base: list[dict] | None, gold: list[dict], nli: dict,
                rel: dict, per_group: int = 10, seed: int = 42) -> list[dict]:
    """Sample up to ``per_group`` rows per failure group (see module docstring)."""
    by_id = {r["id"]: r for r in gold}
    pred = {p["id"]: p for p in preds}
    base_pred = {p["id"]: p["pred_label"] for p in base} if base else {}
    candidates: dict[str, list[dict]] = {g: [] for g in GROUPS}
    for rid, p in pred.items():
        rec = by_id.get(rid)
        if rec is None:
            continue
        orig_id = split_twin_id(rid)[0]
        orig_pred = pred.get(orig_id, {}).get("pred_label") if orig_id != rid else None
        for g in group_of(rid, rec["label"], p["pred_label"], orig_pred, base_pred.get(rid)):
            # top-2 snippets by relevance (twins fall back to their source's scores)
            rel_rows = rel.get(rid) or rel.get(orig_id) or {}
            order = sorted(range(len(rec["evidence"])), key=lambda i: -rel_rows.get(i, (0,))[0])
            snips = [snippet_text(rec, i, nli) for i in order[:2]] + [("", "")] * 2
            candidates[g].append({
                "group": g, "id": rid, "twin_type": twin_type(rid) or "", "gold": rec["label"],
                "pred": p["pred_label"],
                **{f"p_{k}": round(v, 3) for k, v in p["probs"].items()},
                "base_pred": base_pred.get(rid, ""), "claim": rec["claim"],
                "orig_claim": by_id[orig_id]["claim"] if orig_id != rid and orig_id in by_id else "",
                "snippet_1": snips[0][0], "nli_1": snips[0][1],
                "snippet_2": snips[1][0], "nli_2": snips[1][1], "cause": ""})
    rng = random.Random(seed)
    rows = []
    for g in GROUPS:
        pool = sorted(candidates[g], key=lambda r: r["id"])
        rows += rng.sample(pool, min(per_group, len(pool)))
    return rows


def main() -> None:
    """CLI: write the error-case sheet and print counts per group."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", default="results/H5/preds_twin_s42.jsonl")
    ap.add_argument("--base-preds", default="results/H5/preds_notwin_s42.jsonl")
    ap.add_argument("--gold", nargs="+", default=["data/processed/en/multifc_test.jsonl",
                                                 "data/copies/en/test.jsonl"])
    ap.add_argument("--nli", nargs="+", default=["features/nli_zeroshot/multifc_test.parquet",
                                                "features/nli_zeroshot/copies_en/test.parquet"])
    ap.add_argument("--relevance", nargs="+", default=["features/relevance_en.parquet",
                                                      "features/relevance_copies_en.parquet"])
    ap.add_argument("--per-group", type=int, default=10)
    ap.add_argument("--out", default="results/H6/error_cases.csv")
    args = ap.parse_args()
    gold = [r for p in args.gold for r in load_jsonl(p)]
    rows = build_cases(load_preds(args.preds),
                       load_preds(args.base_preds) if args.base_preds else None, gold,
                       per_snippet(args.nli, ["p_entail", "p_contradict", "p_neutral"]),
                       per_snippet(args.relevance, ["relevance"]), args.per_group)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {g: sum(r["group"] == g for r in rows) for g in GROUPS}
    print(f"wrote {len(rows)} cases to {out}: {counts}")


if __name__ == "__main__":
    main()
