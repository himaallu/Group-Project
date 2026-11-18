"""Make the English twins _P, _B, _C for 500 test + 3,000 train sources (R3).

What it does
    1. Picks the sources (``sources.py``): supported claims with >= 2 snippets of
       relevance > 0.3. Test from MultiFC test (AVeriTeC test tops up if needed),
       train from MultiFC train. Seed 42.
    2. ``_B`` evidence swap (``swap.py``), names from the train split's entities.
    3. ``_C`` unrelated evidence (``unrelated.py``), donors from the same split,
       checked with the fine-tuned reranker.
    4. ``_P`` paraphrase (``paraphrase.py``), Qwen3-8B 4-bit + spaCy entity check.
       Paraphrases are cached in ``<out-dir>/staging/`` so a rerun after a Colab
       disconnect does not redo them.
    Twins keep the split of their source. Sources without a valid twin of some type
    are counted in the report, not hidden.

Inputs
    data/processed/en/{multifc_train,multifc_test,averitec_test}.jsonl   (R1)
    features/relevance_en.parquet, features/entities_en.jsonl            (R2)
    reranker: models/reranker_v1 or its Hugging Face id                  (R2)

Outputs
    data/copies/en/{test,train}.jsonl   twins in the shared record format
    data/copies/en/sources.csv          the chosen sources
    data/copies/en/twin_log.csv         what each twin changed (swap old/new, donor, tries)
    results/R3/make_twins_report.json   counts per type and split, drop reasons

How to run (Colab GPU, from the repo root)
    python -m src.copies.make_twins --n-test 5 --n-train 5 --out-dir data/copies/en/test10 \
        --report data/copies/en/test10/report.json            # 10-row check
    python -m src.copies.make_twins --reranker <hub id>       # full run (~1 h on a T4)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.copies.paraphrase import make_paraphrase_twins
from src.copies.sources import select_sources, write_sources
from src.copies.swap import build_pools, make_swap_twin
from src.copies.unrelated import make_unrelated_twin
from src.eval.loaders import load_jsonl

EN = "data/processed/en"
LOG_COLUMNS = ("id", "type", "split", "kind", "old", "new", "n_snippets_changed",
               "donor_id", "max_relevance", "tries", "attempts")
TYPE_ORDER = {"P": 0, "B": 1, "C": 2}


def load_relevance(path: str) -> dict:
    """{id: {snippet_idx: relevance}} from R2's parquet file."""
    import pandas as pd

    table = pd.read_parquet(path)
    lookup = {}
    for rec_id, idx, rel in zip(table["id"], table["snippet_idx"], table["relevance"]):
        lookup.setdefault(rec_id, {})[int(idx)] = float(rel)
    return lookup


def load_entities(path: str) -> dict:
    """{id: entity row} from R2's entities JSONL file."""
    return {row["id"]: row for row in load_jsonl(path)}


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write records as JSON Lines (UTF-8, Arabic kept readable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def paraphrases_cached(sources: list[dict], split: str, cache_dir: Path, generate, nlp):
    """Run ``make_paraphrase_twins`` once per split and cache the result on disk."""
    cache = cache_dir / f"paraphrase_{split}_{len(sources)}.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        print(f"  using cached paraphrases {cache}")
        return [tuple(pair) for pair in saved["pairs"]], Counter(saved["stats"])
    pairs, stats = make_paraphrase_twins(sources, generate, nlp)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"pairs": pairs, "stats": stats}, ensure_ascii=False),
                     encoding="utf-8")
    return pairs, stats


def make_split(split: str, sources: list[dict], donors: list[dict], entities: dict,
               pools: dict, score, generate, nlp, cache_dir: Path, seed: int):
    """Make all three twin types for one split.

    Returns:
        (twins, log rows, stats Counter)
    """
    twins, logs, stats = [], [], Counter()
    donors = [d for d in donors if d["evidence"]]
    for rec in sources:
        for kind, result in (("B", make_swap_twin(rec, entities[rec["id"]], pools, seed)),
                             ("C", make_unrelated_twin(rec, donors, entities, score, seed))):
            if result is None:
                stats[f"{kind}_no_twin"] += 1
            else:
                twins.append(result[0])
                logs.append(result[1])
                stats[f"{kind}_made"] += 1
    if generate is not None:
        pairs, p_stats = paraphrases_cached(sources, split, cache_dir, generate, nlp)
        for twin, log in pairs:
            twins.append(twin)
            logs.append(log)
        stats["P_made"] = len(pairs)
        stats.update({f"P_{k}": v for k, v in p_stats.items()})
    # Group twins by source id, in the order P, B, C.
    twins.sort(key=lambda t: (t["id"][:-2], TYPE_ORDER[t["id"][-1]]))
    return twins, logs, stats


def main() -> None:
    """Parse arguments, build twins for test and train, write all outputs."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--relevance", default="features/relevance_en.parquet")
    ap.add_argument("--entities", default="features/entities_en.jsonl")
    ap.add_argument("--reranker", default="models/reranker_v1")
    ap.add_argument("--spacy-model", default="en_core_web_trf")
    ap.add_argument("--out-dir", default="data/copies/en")
    ap.add_argument("--report", default="results/R3/make_twins_report.json")
    ap.add_argument("--skip-paraphrase", action="store_true", help="only _B and _C")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = {name: load_jsonl(f"{EN}/{name}.jsonl")
            for name in ("multifc_train", "multifc_test", "averitec_test")}
    relevance = load_relevance(args.relevance)
    entities = load_entities(args.entities)
    pools = build_pools([entities[r["id"]] for r in data["multifc_train"]])
    print(f"name pools: {len(pools['place'])} places, {len(pools['person'])} people")

    sources = {
        "test": select_sources([data["multifc_test"], data["averitec_test"]],
                               relevance, args.n_test, args.seed),
        "train": select_sources([data["multifc_train"]], relevance, args.n_train, args.seed),
    }
    out_dir = Path(args.out_dir)
    write_sources(sources["test"] + sources["train"], out_dir / "sources.csv")

    from src.retrieval.rerank import Reranker
    reranker = Reranker(args.reranker)
    generate, nlp = None, None
    if not args.skip_paraphrase:
        from src.copies.paraphrase import Paraphraser
        from src.data.entities import load_nlp
        nlp = load_nlp(args.spacy_model)
        generate = Paraphraser().generate

    by_id = {r["id"]: r for records in data.values() for r in records}
    donor_pools = {"test": data["multifc_test"] + data["averitec_test"],
                   "train": data["multifc_train"]}
    report, all_logs = {}, []
    for split in ("test", "train"):
        source_records = [by_id[row["id"]] for row in sources[split]]
        print(f"{split}: {len(source_records)} sources "
              f"({Counter(r['source'] for r in source_records)})")
        twins, logs, stats = make_split(split, source_records, donor_pools[split], entities,
                                        pools, reranker.score, generate, nlp,
                                        out_dir / "staging", args.seed)
        write_jsonl(twins, out_dir / f"{split}.jsonl")
        all_logs += logs
        report[split] = {"n_sources": len(source_records),
                         "sources_by_dataset": Counter(r["source"] for r in source_records),
                         "n_twins": len(twins), **stats}
        print(f"  {len(twins)} twins -> {out_dir / f'{split}.jsonl'}; {dict(stats)}")

    with open(out_dir / "twin_log.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(all_logs)
    report["swap_kinds"] = Counter(log["kind"] for log in all_logs if log["type"] == "B")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
