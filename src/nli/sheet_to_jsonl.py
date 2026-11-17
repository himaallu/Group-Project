"""Turn a hand-labelled gold sheet into ``data/nli/gold_val.jsonl`` (task H2).

What it does
    Reads the CSV made by ``src/nli/sample_gold.py`` after the ``label`` column has
    been filled in, checks every label, and writes the gold file. Rows already in
    the gold file for OTHER languages (e.g. Youssef's Arabic rows) are kept; rows
    for the sheet's language are replaced.

Inputs
    Labelled CSV (columns ``id, claim_id, snippet_idx, language, claim, snippet,
    label, note``). Labels: ``entail``, ``contradict``, ``neutral`` (or e / c / n,
    any case).

Outputs
    ``data/nli/gold_val.jsonl``, one record per pair:
    ``{"id", "claim_id", "snippet_idx", "language", "claim", "snippet", "label"}``
    with ``label`` in ``entail | contradict | neutral``.

How to run
    python -m src.nli.sheet_to_jsonl results/H2/gold_sheet_en.csv --out data/nli/gold_val.jsonl
    (add ``--allow-partial`` to write only the rows labelled so far)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.eval.loaders import load_jsonl

NLI_LABELS = ("entail", "contradict", "neutral")
SHORTCUTS = {"e": "entail", "c": "contradict", "n": "neutral",
             "entailment": "entail", "contradiction": "contradict"}
GOLD_FIELDS = ("id", "claim_id", "snippet_idx", "language", "claim", "snippet", "label")


def normalise_label(raw: str) -> str | None:
    """Map a typed label to ``entail | contradict | neutral``.

    Returns:
        The normalised label, ``""`` for an empty cell, or None if unrecognised.
    """
    text = (raw or "").strip().lower()
    if not text:
        return ""
    text = SHORTCUTS.get(text, text)
    return text if text in NLI_LABELS else None


def read_sheet(path: Path, allow_partial: bool = False) -> list[dict]:
    """Read and check a labelled sheet.

    Args:
        path: labelled CSV.
        allow_partial: if True, unlabelled rows are skipped; otherwise they are an error.

    Returns:
        Gold records with the fields in ``GOLD_FIELDS``.

    Raises:
        ValueError: listing every row with a bad label (or a missing one, unless
            ``allow_partial``), or duplicate ids.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    records, errors, seen = [], [], set()
    for line_no, row in enumerate(rows, start=2):  # line 1 is the header
        label = normalise_label(row.get("label", ""))
        if label is None:
            errors.append(f"line {line_no} ({row['id']}): unknown label '{row['label']}'")
            continue
        if label == "":
            if not allow_partial:
                errors.append(f"line {line_no} ({row['id']}): no label")
            continue
        if row["id"] in seen:
            errors.append(f"line {line_no}: duplicate id {row['id']}")
            continue
        seen.add(row["id"])
        records.append({
            "id": row["id"], "claim_id": row["claim_id"],
            "snippet_idx": int(row["snippet_idx"]), "language": row["language"],
            "claim": row["claim"], "snippet": row["snippet"], "label": label,
        })
    if errors:
        raise ValueError(f"{len(errors)} problem(s) in {path}:\n  " + "\n  ".join(errors))
    return records


def merge_into_gold(new_records: list[dict], gold_path: Path) -> list[dict]:
    """Combine new records with an existing gold file.

    Existing rows in other languages are kept; existing rows in the new records'
    language(s) are replaced.

    Args:
        new_records: output of :func:`read_sheet`.
        gold_path: existing gold file (may not exist yet).

    Returns:
        All records to write, existing other-language rows first.
    """
    new_languages = {r["language"] for r in new_records}
    existing = load_jsonl(gold_path) if gold_path.exists() else []
    kept = [r for r in existing if r.get("language") not in new_languages]
    return kept + new_records


def write_gold(records: list[dict], path: Path) -> None:
    """Write gold records as JSON Lines (Arabic kept readable, not escaped)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({k: rec[k] for k in GOLD_FIELDS}, ensure_ascii=False) + "\n")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Labelled gold sheet -> gold_val.jsonl")
    parser.add_argument("sheet", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/nli/gold_val.jsonl"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    new_records = read_sheet(args.sheet, args.allow_partial)
    all_records = merge_into_gold(new_records, args.out)
    write_gold(all_records, args.out)
    counts = Counter((r["language"], r["label"]) for r in all_records)
    print(f"wrote {len(all_records)} pairs to {args.out} ({len(new_records)} from {args.sheet})")
    for (lang, label), c in sorted(counts.items()):
        print(f"  {lang} {label}: {c}")


if __name__ == "__main__":
    main()
