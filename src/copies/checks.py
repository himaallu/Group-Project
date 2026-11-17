"""Automatic checks on the English twin files (R3).

What it does
    For every twin in data/copies/en/{test,train}.jsonl it checks:
      all     valid shared format (``loaders.validate_record``), unique id, source
              exists, same split and source as the original, expected label
              (_B contradicted, _C unverified, _P same as original)
      _P      evidence identical to the original, claim text different
      _B      claim identical; evidence equals the original with exactly the logged
              old -> new replacement; at least one snippet changed
      _C      claim identical; evidence equals the logged donor's evidence; donor
              max relevance < 0.1; no shared entity with the donor
    Writes 10 examples of each twin type for the group post.

Inputs
    --twins data/copies/en/test.jsonl data/copies/en/train.jsonl
    --log data/copies/en/twin_log.csv  --entities features/entities_en.jsonl

Outputs
    results/R3/checks.json and results/R3/samples/{P,B,C}_10.jsonl

How to run (laptop or Colab)
    python -m src.copies.checks
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.copies.swap import mention_pattern
from src.copies.unrelated import MAX_DONOR_RELEVANCE, entity_texts
from src.eval.loaders import load_jsonl, split_twin_id, validate_record

EN = "data/processed/en"
ORIGINAL_FILES = [f"{EN}/{d}_{s}.jsonl" for d in ("multifc", "averitec")
                  for s in ("train", "val", "test")]
EXPECTED = {"B": "contradicted", "C": "unverified"}


def texts(record: dict) -> list[str]:
    """Snippet texts of a record."""
    return [ev["text"] for ev in record["evidence"]]


def check_twin(twin: dict, originals: dict, logs: dict, entities: dict | None) -> list[str]:
    """Problems with one twin (empty list = passes every check)."""
    problems = validate_record(twin)
    base, suffixes = split_twin_id(twin["id"])
    kind = suffixes[-1] if suffixes else None
    orig = originals.get(base)
    if orig is None or kind not in ("P", "B", "C"):
        return problems + ["source record or twin type not found"]
    if twin["split"] != orig["split"] or twin["source"] != orig["source"]:
        problems.append("split/source differ from the original")
    if twin["label"] != EXPECTED.get(kind, orig["label"]):
        problems.append(f"label {twin['label']} is not the expected one")
    log = logs.get(twin["id"], {})

    if kind == "P":
        if twin["evidence"] != orig["evidence"]:
            problems.append("_P evidence changed")
        if twin["claim"].strip().lower() == orig["claim"].strip().lower():
            problems.append("_P claim not reworded")
    elif kind == "B":
        if twin["claim"] != orig["claim"]:
            problems.append("_B claim changed")
        pattern = mention_pattern(log.get("old", ""))
        new = log.get("new", "")
        expected = [pattern.sub(lambda _: new, t) for t in texts(orig)]
        if not log or texts(twin) != expected:
            problems.append("_B evidence differs from the logged swap")
        if texts(twin) == texts(orig):
            problems.append("_B no snippet changed")
    elif kind == "C":
        donor = originals.get(log.get("donor_id"))
        if twin["claim"] != orig["claim"]:
            problems.append("_C claim changed")
        if donor is None or twin["evidence"] != donor["evidence"]:
            problems.append("_C evidence is not the logged donor's")
        if float(log.get("max_relevance") or 1) >= MAX_DONOR_RELEVANCE:
            problems.append("_C donor relevance >= 0.1")
        if entities and donor is not None and \
                entity_texts(entities[base], claim_only=True) & entity_texts(entities[donor["id"]]):
            problems.append("_C shares an entity with the donor")
    return problems


def run_checks(twin_files: list[str], originals: dict, logs: dict,
               entities: dict | None) -> tuple[dict, dict]:
    """Check every twin file.

    Returns:
        (report dict, {twin type: first 10 twins}) for the samples.
    """
    report, samples, seen = {}, {"P": [], "B": [], "C": []}, set()
    for path in twin_files:
        twins = load_jsonl(path)
        counts, failures = Counter(), []
        for twin in twins:
            kind = twin["id"].rsplit("_", 1)[-1]
            counts[kind] += 1
            problems = check_twin(twin, originals, logs, entities)
            if twin["id"] in seen:
                problems.append("duplicate id")
            seen.add(twin["id"])
            if problems:
                failures.append({"id": twin["id"], "problems": problems})
            elif kind in samples and len(samples[kind]) < 10:
                samples[kind].append(twin)
        report[path] = {"n_twins": len(twins), "per_type": dict(counts),
                        "n_failed": len(failures), "failures": failures[:50]}
    return report, samples


def main() -> None:
    """CLI: run the checks and write the report and samples."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--twins", nargs="+",
                    default=["data/copies/en/test.jsonl", "data/copies/en/train.jsonl"])
    ap.add_argument("--originals", nargs="+", default=ORIGINAL_FILES)
    ap.add_argument("--log", default="data/copies/en/twin_log.csv")
    ap.add_argument("--entities", default="features/entities_en.jsonl")
    ap.add_argument("--out", default="results/R3/checks.json")
    ap.add_argument("--samples-dir", default="results/R3/samples")
    args = ap.parse_args()

    originals = {r["id"]: r for path in args.originals for r in load_jsonl(path)}
    with open(args.log, encoding="utf-8") as f:
        logs = {row["id"]: row for row in csv.DictReader(f)}
    entities = None
    if Path(args.entities).exists():
        entities = {row["id"]: row for row in load_jsonl(args.entities)}
    report, samples = run_checks(args.twins, originals, logs, entities)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for kind, rows in samples.items():
        out = Path(args.samples_dir) / f"{kind}_10.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                       encoding="utf-8")
    for path, rep in report.items():
        print(f"{path}: {rep['n_twins']} twins {rep['per_type']}, {rep['n_failed']} failed")


if __name__ == "__main__":
    main()
