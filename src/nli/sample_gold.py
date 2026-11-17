"""Sample (claim, snippet) pairs for hand-labelled NLI gold data (task H2).

What it does
    Picks N claims from a shared-format file (default: equal numbers per claim label,
    seed 42), takes ONE snippet per claim, and writes a CSV sheet with an empty
    ``label`` column to fill in by hand. The claim's verdict is NOT shown in the
    sheet, so it cannot bias the label.

    Snippet choice: by default the snippet with the highest word overlap with the
    claim (ties broken at random). Random snippets are mostly unrelated, which would
    give an almost all-"neutral" gold set; overlap gives more entail/contradict pairs.
    Use ``--pick random`` for purely random snippets.

Inputs
    A shared-format ``.jsonl`` file, normally ``data/processed/en/multifc_val.jsonl``
    (gold pairs come from val, never test).

Outputs
    CSV with columns ``id, claim_id, snippet_idx, language, claim, snippet, label,
    note`` (UTF-8 with BOM so Excel / Google Sheets show Arabic correctly).

How to run
    python -m src.nli.sample_gold data/processed/en/multifc_val.jsonl \\
        --n 150 --out results/H2/gold_sheet_en.csv

Labelling guide (fill the ``label`` column; e / c / n are accepted as shortcuts)
    entail      the snippet alone shows the claim is true
    contradict  the snippet alone shows the claim is false
    neutral     the snippet is unrelated, or related but does not settle the claim
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

from src.eval.loaders import LABELS, load_jsonl

SHEET_COLUMNS = ["id", "claim_id", "snippet_idx", "language", "claim", "snippet",
                 "label", "note"]
WORD = re.compile(r"\w+", re.UNICODE)


def word_overlap(claim: str, snippet: str) -> float:
    """Share of the claim's distinct words (lower-cased) that appear in the snippet."""
    claim_words = set(WORD.findall(claim.lower()))
    if not claim_words:
        return 0.0
    return len(claim_words & set(WORD.findall(snippet.lower()))) / len(claim_words)


def pick_snippet(record: dict, rng: random.Random, how: str = "overlap") -> int | None:
    """Choose one snippet index for a claim.

    Args:
        record: shared-format record.
        rng: random generator (for ties and ``how="random"``).
        how: ``"overlap"`` (most word overlap with the claim) or ``"random"``.

    Returns:
        Index into ``record["evidence"]``, or None if there is no non-empty snippet.
    """
    candidates = [i for i, s in enumerate(record.get("evidence") or [])
                  if str(s.get("text", "")).strip()]
    if not candidates:
        return None
    if how == "random":
        return rng.choice(candidates)
    scores = {i: word_overlap(record["claim"], record["evidence"][i]["text"]) for i in candidates}
    best = max(scores.values())
    return rng.choice([i for i in candidates if scores[i] == best])


def sample_claims(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Pick n claims with (as near as possible) equal numbers per label.

    If a label has too few claims, the remaining places are filled at random from
    the other labels.

    Args:
        records: shared-format records (only those with snippets are used).
        n: number of claims to pick.
        rng: random generator.

    Returns:
        The chosen records, shuffled.
    """
    usable = [r for r in records if r.get("evidence")]
    by_label = {lab: [r for r in usable if r["label"] == lab] for lab in LABELS}
    per_label = n // len(LABELS)
    chosen = []
    for lab in LABELS:
        pool = by_label[lab][:]
        rng.shuffle(pool)
        chosen += pool[:per_label]
    # Fill what is left (rounding, or labels with too few claims) from the rest.
    chosen_ids = {r["id"] for r in chosen}
    rest = [r for r in usable if r["id"] not in chosen_ids]
    rng.shuffle(rest)
    chosen += rest[: max(0, n - len(chosen))]
    rng.shuffle(chosen)
    return chosen[:n]


def build_sheet(records: list[dict], n: int = 150, seed: int = 42,
                how: str = "overlap") -> list[dict]:
    """Build the rows of the labelling sheet.

    Args:
        records: shared-format records.
        n: number of pairs.
        seed: random seed (42).
        how: snippet choice, ``"overlap"`` or ``"random"``.

    Returns:
        List of row dicts with the keys in ``SHEET_COLUMNS`` (``label`` and
        ``note`` empty). Pair ids are ``<claim id>_s<snippet index>``.
    """
    rng = random.Random(seed)
    rows = []
    for rec in sample_claims(records, n, rng):
        idx = pick_snippet(rec, rng, how)
        if idx is None:
            continue
        rows.append({
            "id": f"{rec['id']}_s{idx}",
            "claim_id": rec["id"],
            "snippet_idx": idx,
            "language": rec["language"],
            "claim": rec["claim"],
            "snippet": rec["evidence"][idx]["text"],
            "label": "",
            "note": "",
        })
    return rows


def write_sheet(rows: list[dict], path: Path) -> None:
    """Write sheet rows to CSV (UTF-8 with BOM for spreadsheet apps)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Sample NLI gold pairs for hand labelling.")
    parser.add_argument("input", type=Path, help="shared-format jsonl (val split)")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pick", choices=["overlap", "random"], default="overlap")
    parser.add_argument("--out", type=Path, default=Path("results/H2/gold_sheet_en.csv"))
    args = parser.parse_args()

    records = load_jsonl(args.input)
    splits = {r.get("split") for r in records}
    if "test" in splits:
        raise SystemExit("refusing to sample gold pairs from test data")
    rows = build_sheet(records, args.n, args.seed, args.pick)
    write_sheet(rows, args.out)
    print(f"wrote {len(rows)} pairs to {args.out} (from {len(records)} claims, splits {splits})")


if __name__ == "__main__":
    main()
