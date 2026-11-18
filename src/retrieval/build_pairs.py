"""Build (claim, snippet) training pairs for the relevance reranker (R2).

What it does
    For each sampled claim it makes two pairs:
        relevant   (label 1): the claim + one of its own snippets
        irrelevant (label 0): the claim + a snippet of a different, random claim
    Claims are sampled with a fixed seed, so the same pairs come out every time.
    ``train_reranker.py`` calls ``make_pairs`` directly; this CLI only writes a
    small sample so people can look at the pairs.

Inputs
    JSONL files in the shared record format (data/processed/en/*_{train,val}.jsonl).

Outputs
    A JSONL file of pairs: {"claim_id", "claim", "snippet", "snippet_from", "label"}
    (``snippet_from`` = id of the record the snippet came from).

How to run (from the repo root)
    python -m src.retrieval.build_pairs data/processed/en/multifc_train.jsonl \
        --n-claims 5 --out results/R2/samples/pairs_train_10rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.eval.loaders import load_jsonl

SEED = 42


def load_records(paths: list[str | Path]) -> list[dict]:
    """Read several shared-format files and keep only records with snippets.

    Args:
        paths: JSONL files.

    Returns:
        Records (in file order) that have at least one non-empty snippet.
    """
    records = []
    for path in paths:
        for rec in load_jsonl(path):
            if any((ev.get("text") or "").strip() for ev in rec.get("evidence") or []):
                records.append(rec)
    return records


def snippet_texts(record: dict) -> list[str]:
    """Non-empty snippet texts of one record."""
    return [ev["text"].strip() for ev in record.get("evidence") or []
            if (ev.get("text") or "").strip()]


def make_pairs(records: list[dict], n_claims: int | None = None,
               seed: int = SEED) -> list[dict]:
    """Make one relevant and one irrelevant pair per sampled claim.

    Args:
        records: shared-format records, each with at least one snippet
            (use ``load_records``). Needs at least 2 records.
        n_claims: how many claims to sample (None or more than available = all).
        seed: random seed (42 in every run we report).

    Returns:
        List of pair dicts, 2 per claim, shuffled.
    """
    if len(records) < 2:
        raise ValueError("need at least 2 records to draw irrelevant snippets")
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    chosen = order[:n_claims] if n_claims else order

    pairs = []
    for i in chosen:
        rec = records[i]
        own = snippet_texts(rec)
        pairs.append({"claim_id": rec["id"], "claim": rec["claim"],
                      "snippet": rng.choice(own), "snippet_from": rec["id"], "label": 1})
        # Irrelevant snippet: from a different claim whose text differs from this claim's
        # (so duplicates of the same claim never count as negatives).
        while True:
            other = records[rng.randrange(len(records))]
            if other["id"] != rec["id"] and other["claim"].strip().lower() != \
                    rec["claim"].strip().lower():
                break
        pairs.append({"claim_id": rec["id"], "claim": rec["claim"],
                      "snippet": rng.choice(snippet_texts(other)),
                      "snippet_from": other["id"], "label": 0})
    rng.shuffle(pairs)
    return pairs


def binary_auroc(labels: list[int], scores: list[float]) -> float | None:
    """Area under the ROC curve for 0/1 labels (rank formula, ties count half).

    Args:
        labels: 0/1 gold labels.
        scores: higher = more likely 1.

    Returns:
        AUROC in [0, 1], or None if only one class is present.
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    # Rank all scores once; average ranks for ties (Mann-Whitney U statistic).
    ordered = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and scores[ordered[end + 1]] == scores[ordered[start]]:
            end += 1
        for k in range(start, end + 1):
            ranks[ordered[k]] = (start + end) / 2 + 1
        start = end + 1
    pos_rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (pos_rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    """CLI: write a pair file (or a small sample) for inspection."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="shared-format JSONL files")
    ap.add_argument("--n-claims", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pairs = make_pairs(load_records(args.files), args.n_claims, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"{len(pairs)} pairs ({sum(p['label'] for p in pairs)} relevant) -> {args.out}")


if __name__ == "__main__":
    main()
