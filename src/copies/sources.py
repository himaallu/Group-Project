"""Pick the source claims that twins are made from (R3, CLAUDE.md §4 "Twin rules").

What it does
    A claim can be a twin source if it is ``supported`` and has at least 2 snippets
    with reranker relevance > 0.3 (from R2's ``features/relevance_en.parquet``).
    From the eligible claims we take 500 test and 3,000 train sources, fixed with
    seed 42. If the main test file has fewer than 500 eligible claims, the next test
    file (AVeriTeC) tops it up; the shortfall is reported, never hidden.

Inputs
    Records in the shared format and a relevance lookup {id: {snippet_idx: relevance}}.

Outputs
    ``data/copies/en/sources.csv`` with columns id, split, source, n_relevant.

How to run
    Called by ``src/copies/make_twins.py``; there is no separate CLI.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

RELEVANCE_MIN = 0.3
MIN_RELEVANT_SNIPPETS = 2
SOURCE_COLUMNS = ("id", "split", "source", "n_relevant")


def n_relevant(record: dict, relevance: dict, threshold: float = RELEVANCE_MIN) -> int:
    """How many of the record's snippets have relevance above ``threshold``.

    Args:
        record: shared-format record.
        relevance: {id: {snippet_idx: relevance}}; missing snippets count as 0.
    """
    scores = relevance.get(record["id"], {})
    return sum(1 for idx in range(len(record.get("evidence") or []))
               if scores.get(idx, 0.0) > threshold)


def is_eligible(record: dict, relevance: dict) -> bool:
    """True if the record is a supported claim with >= 2 relevant snippets."""
    return record["label"] == "supported" and \
        n_relevant(record, relevance) >= MIN_RELEVANT_SNIPPETS


def select_sources(groups: list[list[dict]], relevance: dict, n: int,
                   seed: int = 42) -> list[dict]:
    """Choose ``n`` eligible sources, using the groups in order of preference.

    Each group's eligible records are sorted by id and shuffled with ``seed``; the
    first group is used up before the next one is touched.

    Args:
        groups: lists of records, e.g. [multifc_test, averitec_test].
        relevance: {id: {snippet_idx: relevance}}.
        n: how many sources to return (fewer if not enough are eligible).
        seed: random seed.

    Returns:
        Rows {"id", "split", "source", "n_relevant"} for the chosen records.
    """
    rng = random.Random(seed)
    chosen = []
    for records in groups:
        eligible = sorted((r for r in records if is_eligible(r, relevance)),
                          key=lambda r: r["id"])
        rng.shuffle(eligible)
        for rec in eligible[:n - len(chosen)]:
            chosen.append({"id": rec["id"], "split": rec["split"], "source": rec["source"],
                           "n_relevant": n_relevant(rec, relevance)})
        if len(chosen) >= n:
            break
    return chosen


def write_sources(rows: list[dict], path: str | Path) -> None:
    """Write the sources list as CSV (``SOURCE_COLUMNS``)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
