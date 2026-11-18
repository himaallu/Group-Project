"""Convert the Arabic part of X-FACT to the TwinCheck shared record format (task Y1).

What it does
    Reads X-FACT's train/dev/test TSV files (GitHub utahnlp/x-fact, folder
    data/x-fact/), keeps rows with language == "ar", maps X-FACT's coarse labels to
    supported / contradicted / unverified (dropping the rest), turns evidence_1..5
    and link_1..5 into the shared evidence list, and writes one JSONL file per split.
    X-FACT has real labels for its own splits, so we keep them (dev -> "val").
    ood.tsv and zeroshot.tsv contain no Arabic, so they are not read.

Inputs
    --raw_dir   folder with train.all.tsv, dev.all.tsv, test.all.tsv
    --limit     only the first N Arabic rows of each split (use 10 for the test run)

Outputs
    <out_dir>/xfact_ar_train.jsonl, xfact_ar_val.jsonl, xfact_ar_test.jsonl
    These are uncleaned; Rathi's src/data/clean.py writes the final
    data/processed/ar/ files.

How to run
    git clone --depth 1 https://github.com/utahnlp/x-fact data/raw/x-fact
    python src/data/load_xfact.py --raw_dir data/raw/x-fact/data/x-fact \
        --out_dir data/processed/ar/uncleaned --limit 10
"""

import argparse
import csv
import json
import os
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

# X-FACT already maps every site's verdict to 7 coarse English labels.
# This follows the team label map in CLAUDE.md section 4 (same rows go in data/label_map.csv).
LABEL_MAP = {
    "true": "supported",
    "mostly true": "supported",
    "false": "contradicted",
    "mostly false": "contradicted",
    "complicated/hard to categorise": "unverified",  # includes "unproven", "unverifiable"
    "partly true/misleading": None,  # dropped (half true / misleading)
    "other": None,                   # dropped (satire, myth, clarification ...)
}
SPLITS = {"train": "train.all.tsv", "val": "dev.all.tsv", "test": "test.all.tsv"}
N_EVIDENCE = 5


def read_xfact_arabic(raw_dir, limit=None):
    """Read X-FACT TSVs and keep only Arabic rows.

    Args:
        raw_dir: folder that holds train.all.tsv, dev.all.tsv, test.all.tsv.
        limit: keep only the first `limit` Arabic rows per split (None = all).
    Returns:
        dict split -> DataFrame (all columns as strings, empty string for missing).
    """
    out = {}
    for split, fname in SPLITS.items():
        # QUOTE_NONE: snippets contain quote characters that would otherwise break parsing
        df = pd.read_csv(os.path.join(raw_dir, fname), sep="\t", quoting=csv.QUOTE_NONE,
                         dtype=str, keep_default_na=False)
        df = df[df["language"] == "ar"].reset_index(drop=True)
        out[split] = df.head(limit) if limit else df
    return out


def domain_of(url):
    """Return the domain of a URL without 'www.', or None for empty / NO_LINK values."""
    if not url or url == "NO_LINK":
        return None
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc or None


def build_evidence(row):
    """Turn evidence_1..5 / link_1..5 of one X-FACT row into the shared evidence list."""
    evidence = []
    for i in range(1, N_EVIDENCE + 1):
        text = row.get(f"evidence_{i}", "").strip()
        if not text:
            continue
        url = row.get(f"link_{i}", "").strip()
        url = "" if url == "NO_LINK" else url
        # empty strings (not None) for missing links, as clean.py expects strings
        evidence.append({"text": text, "url": url, "domain": domain_of(url) or ""})
    return evidence


def row_to_record(row, number, split):
    """Convert one X-FACT row to a shared-format record.

    Args:
        row: dict for one TSV row.
        number: running number used in the id (xfact_ar_000001), fixed by file order.
        split: "train", "val" or "test".
    Returns:
        record dict, or None if the label is dropped by the label map.
    """
    orig = row["label"].strip().lower()
    label = LABEL_MAP.get(orig)
    if label is None:
        return None
    return {
        "id": f"xfact_ar_{number:06d}",
        "source": "xfact",
        "split": split,
        "language": "ar",
        "claim": row["claim"].strip(),
        "evidence": build_evidence(row),
        "image_path": None,
        "label": label,
        "orig_label": orig,
    }


def main():
    """Convert all splits, write JSONL files and print counts per split and label."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", default="data/processed/ar/uncleaned")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    number = 0  # numbers run across splits so every id is unique
    for split, df in read_xfact_arabic(args.raw_dir, args.limit).items():
        kept, dropped = [], Counter()
        for row in df.to_dict("records"):
            number += 1
            rec = row_to_record(row, number, split)
            if rec is None:
                dropped[row["label"]] += 1
            else:
                kept.append(rec)
        path = os.path.join(args.out_dir, f"xfact_ar_{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        labels = Counter(r["label"] for r in kept)
        no_ev = sum(1 for r in kept if not r["evidence"])
        print(f"{split}: read={len(df)} kept={len(kept)} {dict(labels)} "
              f"dropped={sum(dropped.values())} {dict(dropped)} no_evidence={no_ev} -> {path}")


if __name__ == "__main__":
    main()
