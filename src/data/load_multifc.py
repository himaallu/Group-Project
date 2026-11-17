"""Convert MultiFC (authors' release with snippets) to the TwinCheck shared format (R1).

What it does
    Reads MultiFC ``train.tsv`` and ``dev.tsv`` plus the Google snippets in
    ``snippets/<claimID>``, maps each site's label with ``data/label_map.csv``,
    and writes one JSONL file in the shared record format. The official
    ``test.tsv`` is ignored: its labels were never released (CLAUDE.md section 4),
    so ``clean.py`` later makes our own 80/10/10 split over train + dev.

    Records whose label maps to "dropped" (half true, satire, ...) or is not in the
    label map are left out and counted.

Inputs
    --raw-dir    folder with train.tsv, dev.tsv and snippets/
                 (default data/raw/multifc/multi_fc_publicdata)
    --label-map  data/label_map.csv
    --limit      only the first N rows of each file (use 10 for the test run)

Outputs
    --out        data/processed/en/staging/multifc_all.jsonl (not split, not cleaned)
    Counts per label and how many claims were dropped are printed.

How to run (from the repo root)
    python -m src.data.load_multifc --limit 10      # 10-row test
    python -m src.data.load_multifc                 # full run

Data source
    multi_fc_publicdata.zip (Augenstein et al., EMNLP 2019; copy shared by
    Hansen et al., 2021): https://www.dropbox.com/s/3v5oy3eddg3506j/multi_fc_publicdata.zip
    The Hugging Face mirror pszemraj/multi_fc has NO snippet text, so we do not use it.
    Licence unknown: research use only, cite the paper.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

from src.data.common import (DROPPED, load_label_map, make_evidence, make_record,
                             map_label, write_jsonl)

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

SOURCE = "multifc"
# Column order of train.tsv / dev.tsv (from the release README).
CLAIM_COLUMNS = ["claimID", "claim", "label", "claimURL", "reason", "categories",
                 "speaker", "checker", "tags", "articleTitle", "publishDate",
                 "claimDate", "entities"]
# Which TSV gives which placeholder split. clean.py re-splits MultiFC 80/10/10.
FILES = {"train": "train.tsv", "val": "dev.tsv"}


def strip_quotes(text: str) -> str:
    """Remove one pair of surrounding double quotes (PolitiFact claims are quoted)."""
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.strip()


def read_claims(path: str, limit: int | None = None) -> list[dict]:
    """Read a MultiFC claims TSV.

    Args:
        path: path to train.tsv or dev.tsv.
        limit: keep only the first ``limit`` rows (None = all).

    Returns:
        List of dicts keyed by ``CLAIM_COLUMNS``.
    """
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        # QUOTE_NONE: fields contain unbalanced quotes; tabs are the only separator.
        for parts in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(parts) < 3:
                continue
            rows.append(dict(zip(CLAIM_COLUMNS, parts)))
            if limit and len(rows) >= limit:
                break
    return rows


def read_snippets(snippet_dir: str, claim_id: str) -> list[dict]:
    """Read the Google snippets for one claim.

    Each line of ``snippets/<claimID>`` is ``rank, title, snippet, url`` (tab-separated).

    Args:
        snippet_dir: the ``snippets/`` folder.
        claim_id: MultiFC claim id, e.g. ``"pomt-03627"``.

    Returns:
        Evidence list in rank order; empty if the claim has no snippet file.
    """
    path = os.path.join(snippet_dir, claim_id)
    if not os.path.exists(path):
        return []
    evidence = []
    with open(path, encoding="utf-8", newline="") as f:
        for parts in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(parts) < 3:
                continue
            text = parts[2].strip()
            url = parts[3].strip() if len(parts) > 3 else ""
            if text:
                evidence.append(make_evidence(text, url))
    return evidence


def convert(raw_dir: str, label_map: dict, limit: int | None = None
            ) -> tuple[list[dict], Counter]:
    """Convert MultiFC train + dev into shared-format records.

    Args:
        raw_dir: folder with train.tsv, dev.tsv and snippets/.
        label_map: output of ``load_label_map``.
        limit: rows per file (None = all).

    Returns:
        (records, stats). ``stats`` counts kept labels and dropped reasons.
    """
    records, stats = [], Counter()
    number = 0  # running number for ids, fixed by file order (train first, then dev)
    snippet_dir = os.path.join(raw_dir, "snippets")
    for split, fname in FILES.items():
        for row in read_claims(os.path.join(raw_dir, fname), limit):
            rec_id = f"{SOURCE}_{number:06d}"
            number += 1
            orig_label = row["label"].strip()
            label = map_label(label_map, SOURCE, orig_label)
            if label is None:
                stats["dropped_unmapped"] += 1
                continue
            if label == DROPPED:
                stats["dropped_by_label_map"] += 1
                continue
            claim = strip_quotes(row["claim"])
            if not claim:
                stats["dropped_empty_claim"] += 1
                continue
            evidence = read_snippets(snippet_dir, row["claimID"])
            records.append(make_record(rec_id, SOURCE, split, claim, evidence,
                                       label, orig_label))
            stats[f"kept_{label}"] += 1
    return records, stats


def main() -> None:
    """Parse arguments, convert MultiFC and write the staging JSONL file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", default=os.path.join(
        "data", "raw", "multifc", "multi_fc_publicdata"))
    parser.add_argument("--label-map", default=os.path.join("data", "label_map.csv"))
    parser.add_argument("--out", default=os.path.join(
        "data", "processed", "en", "staging", "multifc_all.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records, stats = convert(args.raw_dir, load_label_map(args.label_map), args.limit)
    write_jsonl(records, args.out)
    with_snippets = sum(1 for r in records if r["evidence"])
    print(f"MultiFC: kept {len(records)} claims ({with_snippets} with snippets) -> {args.out}")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
