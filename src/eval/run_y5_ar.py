"""E7: our verdict heads and the baselines on Arabic and mixed data (task Y5).

What it does
    Test sets
      - Twin set, 200 test sources (Y4): per language
          en     English original (supported) + its _B (contradicted) + _C (unverified)
          ar     _A (supported) + _B_A (contradicted) + _C_A (unverified)
          mixed  _M (supported: Arabic claim, English evidence)
      - X-FACT Arabic test (135 real Arabic claims, own labels).
      - M4FC Arabic is left out: it has no evidence text, so nothing can be checked
        against evidence (reported in results/Y5/README.md).
    Systems
      - Verdict heads from H5 (zero-shot mDeBERTa NLI + fine-tuned reranker):
        twin, notwin, twin_noent (no entity features), seeds 42/43/44.
        English records use the R2 spaCy entities; Arabic records have none, so their
        entity features are filled with 1.0 (as in src/verdict/infer.py and E5).
        twin_noent never uses entities, so it is the fair cross-language control.
      - Baselines (Feba, F4): zero-shot NLI rule; similarity + LightGBM (seeds 42/43/44).
    Metrics per (dataset, language, system), mean and sd over seeds
      macro_f1   3-class macro-F1 over the records of that language (twin sets:
                 supported / contradicted / unverified come from the original-type,
                 _B-type and _C-type records); not defined for mixed (one class)
      accuracy   share of correct verdicts
      esr        % of _B / _B_A predicted contradicted
      unrelated  % of _C / _C_A predicted unverified
      crossling  % of _A (ar) or _M (mixed) with the same verdict as the English original

Inputs
    data/copies/ar/test.jsonl (Y4), data/copies/en/test.jsonl (R3),
    data/processed/en/multifc_test.jsonl, data/processed/ar/xfact_ar_test.jsonl
    features: nli_zeroshot/{multifc_test, copies_en/test, copies_ar/test, xfact_ar_test}.parquet,
              relevance_en.parquet, relevance_copies_en.parquet, relevance_copies_ar.parquet,
              relevance_ar.parquet, entities_en.jsonl, entities_copies_en.jsonl
    models/H5/verdict_{twin,notwin,twin_noent}_s{42,43,44}/
    baseline predictions in --baseline-dir (preds_zeroshot_nli*.jsonl, preds_similarity*_s*.jsonl)

Outputs
    results/Y5/preds_<system>_s<seed>.jsonl  (verdict-head predictions, all test sets)
    results/Y5/arabic_E7.csv                 one row per dataset x language x system
    results/Y5/arabic_E7_by_seed.csv

How to run (CPU, a few minutes, once the Arabic NLI + relevance files exist)
    python -m src.eval.run_y5_ar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.loaders import join_gold, load_jsonl, split_twin_id
from src.eval.metrics import macro_f1
from src.verdict.features import load_entities
from src.verdict.predict import predict_records
from src.verdict.train import write_jsonl

VARIANTS = ("twin", "notwin", "twin_noent")
SEEDS = (42, 43, 44)
METRICS = ("macro_f1", "accuracy", "esr", "unrelated", "crossling")


def kind(record_id: str) -> str:
    """'orig', 'B', 'C', 'A', 'M', 'B_A' or 'C_A' from a twin id."""
    _, suffixes = split_twin_id(record_id)
    return "_".join(suffixes) if suffixes else "orig"


def twin_set(ar: list[dict], en_twins: list[dict], originals: list[dict]) -> list[dict]:
    """English originals, their _B/_C and the Arabic twins, for the Y4 sources only."""
    bases = {split_twin_id(r["id"])[0] for r in ar}
    en = [r for r in en_twins if split_twin_id(r["id"])[0] in bases and kind(r["id"]) in ("B", "C")]
    orig = [r for r in originals if r["id"] in bases]
    return orig + en + ar


def group_of(record_id: str, dataset: str) -> str:
    """Language group of a record in a dataset."""
    if dataset == "xfact_ar_test":
        return "ar"
    k = kind(record_id)
    return {"orig": "en", "B": "en", "C": "en", "A": "ar", "B_A": "ar", "C_A": "ar", "M": "mixed"}[k]


def scores(df: pd.DataFrame, dataset: str) -> list[dict]:
    """Metric rows per language group for one joined prediction table."""
    df = df.assign(kind=df["id"].map(kind), group=[group_of(i, dataset) for i in df["id"]])
    orig_pred = df[df["kind"] == "orig"].set_index("base_id")["pred"]
    rows = []
    for group, sub in df.groupby("group"):
        b = sub[sub["kind"].isin(["B", "B_A"])]
        c = sub[sub["kind"].isin(["C", "C_A"])]
        lang_twins = sub[sub["kind"].isin(["A", "M"])]
        lang_twins = lang_twins[lang_twins["base_id"].isin(orig_pred.index)]
        rows.append({
            "dataset": dataset, "language": group, "n": len(sub),
            "macro_f1": macro_f1(list(sub["gold"]), list(sub["pred"])) if sub["gold"].nunique() > 1 else None,
            "accuracy": float((sub["gold"] == sub["pred"]).mean()),
            "esr": float((b["pred"] == "contradicted").mean()) if len(b) else None,
            "unrelated": float((c["pred"] == "unverified").mean()) if len(c) else None,
            "crossling": float((lang_twins["pred"].to_numpy() ==
                                orig_pred.loc[lang_twins["base_id"]].to_numpy()).mean())
            if len(lang_twins) else None,
        })
    return rows


def baseline_preds(dirs: list[Path]) -> list[tuple[str, int, list[dict]]]:
    """(system, seed, predictions) for the zero-shot NLI and similarity baselines.

    English predictions come from Feba's results/F4 files, Arabic ones from the
    Y5 baseline folder; files with the same name are merged.
    """
    merged: dict[tuple[str, int], list[dict]] = {}
    for d in dirs:
        for path in sorted(d.glob("preds_*.jsonl")) if d.exists() else []:
            stem = path.stem.removeprefix("preds_")
            if stem == "zeroshot_nli":
                key = ("zeroshot_nli", 0)
            elif stem.startswith("similarity_s"):
                key = ("similarity", int(stem.rsplit("_s", 1)[1]))
            else:
                continue
            merged.setdefault(key, []).extend(load_jsonl(path))
    return [(s, seed, p) for (s, seed), p in merged.items()]


def nli_table(paths: list[str]) -> pd.DataFrame:
    """Concatenate per-snippet NLI parquet files that exist."""
    return pd.concat([pd.read_parquet(p) for p in paths if Path(p).exists()], ignore_index=True)


def summarise(by_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd over seeds per (dataset, language, system)."""
    out = []
    for (dataset, lang, system), g in by_seed.groupby(["dataset", "language", "system"], sort=False):
        row = {"dataset": dataset, "language": lang, "system": system, "n": int(g["n"].iloc[0]),
               "seeds": len(g)}
        for m in METRICS:
            vals = g[m].dropna().astype(float)
            row[f"{m}_mean"] = round(vals.mean(), 4) if len(vals) else None
            row[f"{m}_sd"] = round(vals.std(ddof=0), 4) if len(vals) > 1 else None
        out.append(row)
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Predict with the verdict heads, score them and the baselines, write the E7 tables."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ar-twins", default="data/copies/ar/test.jsonl")
    ap.add_argument("--en-twins", default="data/copies/en/test.jsonl")
    ap.add_argument("--originals", default="data/processed/en/multifc_test.jsonl")
    ap.add_argument("--xfact", default="data/processed/ar/xfact_ar_test.jsonl")
    ap.add_argument("--models-dir", default="models/H5")
    ap.add_argument("--baseline-dir", default="results/Y5/baselines")
    ap.add_argument("--out-dir", default="results/Y5")
    args = ap.parse_args(argv)

    f = Path("features")
    twins = twin_set(load_jsonl(args.ar_twins), load_jsonl(args.en_twins), load_jsonl(args.originals))
    xfact = [r for r in load_jsonl(args.xfact) if r.get("evidence")]
    sets = {"twins": twins, "xfact_ar_test": xfact}
    nli = nli_table([f / "nli_zeroshot/multifc_test.parquet", f / "nli_zeroshot/copies_en/test.parquet",
                     f / "nli_zeroshot/copies_ar/test.parquet", f / "nli_zeroshot/xfact_ar_test.parquet"])
    rel = nli_table([f / "relevance_en.parquet", f / "relevance_copies_en.parquet",
                     f / "relevance_copies_ar.parquet", f / "relevance_ar.parquet"])
    entities = {}
    for p in (f / "entities_en.jsonl", f / "entities_copies_en.jsonl"):
        if p.exists():
            entities.update(load_entities(p))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            model_dir = Path(args.models_dir) / f"verdict_{variant}_s{seed}"
            if not model_dir.exists():
                print(f"skip {model_dir} (missing)")
                continue
            preds = []
            for name, recs in sets.items():
                p = predict_records(model_dir, recs, nli, rel, entities)
                preds += p
                df, _ = join_gold(p, recs)
                rows += [{**r, "system": variant, "seed": seed} for r in scores(df, name)]
            write_jsonl(preds, out / f"preds_{variant}_s{seed}.jsonl")

    for system, seed, preds in baseline_preds([Path("results/F4"), Path(args.baseline_dir)]):
        for name, recs in sets.items():
            ids = {r["id"] for r in recs}
            df, _ = join_gold([p for p in preds if p["id"] in ids], recs)
            if len(df):
                rows += [{**r, "system": f"baseline_{system}", "seed": int(seed)} for r in scores(df, name)]

    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(out / "arabic_E7_by_seed.csv", index=False)
    table = summarise(by_seed)
    table.to_csv(out / "arabic_E7.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table[["dataset", "language", "system", "n"] + [f"{m}_mean" for m in METRICS]])
    return table


if __name__ == "__main__":
    main()
