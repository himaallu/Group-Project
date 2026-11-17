"""Turn a checked review sheet into data/claims/gold.jsonl (task Y3, part 1).

What it does
    Reads the review CSV made by draft_gold.py after Youssef has checked it:
    - edit the message or claim_1..3 cells directly to fix an example;
    - put "n" (or "x") in the ok column to drop an example;
    - anything else in ok (empty, "y") keeps it.
    Kept rows are appended to data/claims/gold.jsonl (ids already there are replaced).

Inputs
    review CSV (results/Y3/review_0001.csv), the draft jsonl (for source_ids)

Outputs
    data/claims/gold.jsonl   {"id", "language", "message", "claims", "source_ids"}

How to run (repo root)
    python -m src.claims.review_to_gold results/Y3/review_0001.csv data/claims/gold_draft_0001.jsonl
"""

from __future__ import annotations

import argparse
import csv
import os

from src.data.common import read_jsonl, write_jsonl

GOLD = "data/claims/gold.jsonl"


def read_review(path: str) -> list[dict]:
    """Read the review sheet (UTF-8 with or without BOM)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_gold(rows: list[dict], drafts: dict) -> tuple[list[dict], int]:
    """Keep checked rows and rebuild gold records; return (records, number dropped)."""
    kept, dropped = [], 0
    for r in rows:
        if r.get("ok", "").strip().lower() in ("n", "x", "no"):
            dropped += 1
            continue
        claims = [r[k].strip() for k in ("claim_1", "claim_2", "claim_3") if r.get(k, "").strip()]
        draft = drafts.get(r["id"], {})
        kept.append({"id": r["id"], "language": r["language"], "message": r["message"].strip(),
                     "claims": claims, "source_ids": draft.get("source_ids", [])})
    return kept, dropped


def main() -> None:
    """Merge the checked examples into data/claims/gold.jsonl and print counts."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review")
    ap.add_argument("draft")
    ap.add_argument("--out", default=GOLD)
    args = ap.parse_args()
    drafts = {d["id"]: d for d in read_jsonl(args.draft)}
    kept, dropped = to_gold(read_review(args.review), drafts)
    existing = read_jsonl(args.out) if os.path.exists(args.out) else []
    new_ids = {r["id"] for r in kept}
    merged = sorted([r for r in existing if r["id"] not in new_ids] + kept, key=lambda r: r["id"])
    write_jsonl(merged, args.out)
    print(f"kept {len(kept)}, dropped {dropped}; {args.out} now has {len(merged)} examples")


if __name__ == "__main__":
    main()
