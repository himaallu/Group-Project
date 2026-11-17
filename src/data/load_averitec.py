"""Convert AVeriTeC to the TwinCheck shared format (R1).

What it does
    Reads AVeriTeC train and dev (downloaded by ``download_en.py``). Each answer in
    a claim's question-answer evidence becomes one snippet:
        "<question>? <answer>"            (extractive / abstractive answers)
        "<question>? <Yes|No>. <reason>"  (boolean answers, with their explanation)
    The question is kept in the text because many answers ("Yes", "In 2019") mean
    nothing on their own. Unanswerable answers are skipped. The snippet URL is the
    answer's source URL.

    Splits: AVeriTeC's test labels are not public, so the official dev set (500)
    becomes our test set and the official train set stays train; ``clean.py``
    (--split keep) then moves 10% of train to val (stratified, seed 42).
    Labels are mapped with ``data/label_map.csv``; "Conflicting Evidence/
    Cherrypicking" is dropped and counted.

Inputs
    --raw-dir    data/raw/averitec (train.json, dev.json)
    --label-map  data/label_map.csv
    --limit      only the first N claims of each file (use 10 for the test run)

Outputs
    --out        data/processed/en/staging/averitec_all.jsonl (not cleaned)

How to run (from the repo root)
    python -m src.data.load_averitec --limit 10
    python -m src.data.load_averitec
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from src.data.common import (DROPPED, load_label_map, make_evidence, make_record,
                             map_label, write_jsonl)

SOURCE = "averitec"
# Official file -> our split (official dev is our test set; val is carved later).
FILES = {"train": "train.json", "test": "dev.json"}


def answer_to_text(question: str, answer: dict) -> str | None:
    """Turn one AVeriTeC answer into snippet text.

    Args:
        question: the question the answer belongs to.
        answer: dict with ``answer``, ``answer_type``, ``boolean_explanation``.

    Returns:
        Snippet text, or None for unanswerable / empty answers.
    """
    text = (answer.get("answer") or "").strip()
    if not text or answer.get("answer_type") == "Unanswerable":
        return None
    question = question.strip()
    if question and not question.endswith("?"):
        question += "?"
    if answer.get("answer_type") == "Boolean" and answer.get("boolean_explanation"):
        text = f"{text.rstrip('.')}. {answer['boolean_explanation'].strip()}"
    return f"{question} {text}".strip()


def build_evidence(claim: dict) -> list[dict]:
    """Build the evidence list for one AVeriTeC claim (one snippet per answer).

    Args:
        claim: one AVeriTeC claim dict.

    Returns:
        Evidence list in the shared format.
    """
    evidence = []
    for qa in claim.get("questions") or []:
        for answer in qa.get("answers") or []:
            text = answer_to_text(qa.get("question", ""), answer)
            if text:
                evidence.append(make_evidence(text, answer.get("source_url") or ""))
    return evidence


def convert(raw_dir: str, label_map: dict, limit: int | None = None
            ) -> tuple[list[dict], Counter]:
    """Convert AVeriTeC train + dev into shared-format records.

    Args:
        raw_dir: folder with train.json and dev.json.
        label_map: output of ``load_label_map``.
        limit: claims per file (None = all).

    Returns:
        (records, stats) where stats counts kept labels and dropped reasons.
    """
    records, stats = [], Counter()
    number = 0  # running number for ids, fixed by file order (train first, then dev)
    for split, fname in FILES.items():
        with open(os.path.join(raw_dir, fname), encoding="utf-8") as f:
            claims = json.load(f)
        for claim in claims[:limit] if limit else claims:
            rec_id = f"{SOURCE}_{number:06d}"
            number += 1
            orig_label = (claim.get("label") or "").strip()
            label = map_label(label_map, SOURCE, orig_label)
            if label is None:
                stats["dropped_unmapped"] += 1
                continue
            if label == DROPPED:
                stats["dropped_by_label_map"] += 1
                continue
            if not (claim.get("claim") or "").strip():
                stats["dropped_empty_claim"] += 1
                continue
            records.append(make_record(rec_id, SOURCE, split, claim["claim"],
                                       build_evidence(claim), label, orig_label))
            stats[f"kept_{split}_{label}"] += 1
    return records, stats


def main() -> None:
    """Parse arguments, convert AVeriTeC and write the staging JSONL file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", default=os.path.join("data", "raw", "averitec"))
    parser.add_argument("--label-map", default=os.path.join("data", "label_map.csv"))
    parser.add_argument("--out", default=os.path.join(
        "data", "processed", "en", "staging", "averitec_all.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records, stats = convert(args.raw_dir, load_label_map(args.label_map), args.limit)
    write_jsonl(records, args.out)
    print(f"AVeriTeC: kept {len(records)} claims -> {args.out}")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
