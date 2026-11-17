"""Build Arabic NLI data from cleaned X-FACT Arabic (task Y2, part 1).

What it does
    1. Gold pool: picks the claims for hand-labelled Arabic gold pairs. Arabic val
       has only 93 claims with snippets, so 57 train claims are added (10 supported,
       7 unverified, 40 contradicted, seed 42) to reach 150. These claims are then
       kept OUT of the training pairs, so gold and train never share a claim.
    2. Weak training pairs from the remaining train claims (same rules as R4):
         supported claim    + its relevant snippet  -> entail
         contradicted claim + its relevant snippet  -> contradict
         any claim          + snippets of 2 random other claims -> neutral
       "Relevant" = the claim's snippets with the most word overlap (at most 2 per
       claim, overlap >= 0.2), since there are no Arabic reranker scores yet.
       Unverified claims give no entail/contradict pairs (their snippets do not
       settle the claim). Contradict and neutral are capped so classes stay balanced;
       entail is topped up later with translated English pairs (translate_pairs.py).

Inputs
    data/processed/ar/xfact_ar_{train,val}.jsonl   (cleaned with src/data/clean.py)

Outputs
    data/nli/ar_gold_pool.jsonl     shared-format claims for src.nli.sample_gold
    data/nli/ar_train_xfact.jsonl   weak pairs, same fields as data/nli/gold_val.jsonl
                                    (id, claim_id, snippet_idx, language, claim, snippet,
                                    label) plus "origin": "xfact_weak"

How to run (repo root)
    python -m src.nli.build_ar_pairs --limit 10          # 10-claim test run
    python -m src.nli.build_ar_pairs
    python -m src.nli.sample_gold data/nli/ar_gold_pool.jsonl --n 150 \
        --out results/Y2/gold_sheet_ar.csv
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter

from src.data.common import SEED, read_jsonl, write_jsonl

WORD = re.compile(r"\w+", re.UNICODE)
EXTRA_TRAIN_FOR_GOLD = {"supported": 10, "unverified": 7, "contradicted": 40}
MAX_SNIPPETS_PER_CLAIM = 2
MIN_OVERLAP = 0.2
NEUTRAL_PER_CLAIM = 2
WEAK_LABEL = {"supported": "entail", "contradicted": "contradict"}


def overlap(claim: str, snippet: str) -> float:
    """Share of the claim's distinct words that also appear in the snippet."""
    cw = set(WORD.findall(claim.lower()))
    return len(cw & set(WORD.findall(snippet.lower()))) / len(cw) if cw else 0.0


def pick_gold_pool(train: list[dict], val: list[dict], rng: random.Random) -> list[dict]:
    """Return val claims plus a fixed, label-stratified sample of train claims.

    Args:
        train, val: cleaned shared-format records (claims with snippets only).
        rng: seeded random generator.
    Returns:
        list of records for the gold pool (val first, then the extra train claims).
    """
    extra = []
    for label, n in EXTRA_TRAIN_FOR_GOLD.items():
        pool = sorted((r for r in train if r["label"] == label), key=lambda r: r["id"])
        extra += rng.sample(pool, min(n, len(pool)))
    return val + extra


def pair(rec: dict, idx: int, label: str, snippet_rec: dict | None = None) -> dict:
    """Make one NLI pair in the gold_val.jsonl format.

    Args:
        rec: the claim's record.
        idx: snippet index inside the record the snippet comes from.
        label: entail / contradict / neutral.
        snippet_rec: record that owns the snippet (another claim for neutral pairs).
    """
    src = snippet_rec or rec
    tag = "s" if snippet_rec is None else f"r_{src['id']}_"
    return {"id": f"{rec['id']}_{tag}{idx}", "claim_id": rec["id"], "snippet_idx": idx,
            "language": "ar", "claim": rec["claim"], "snippet": src["evidence"][idx]["text"],
            "label": label, "origin": "xfact_weak"}


def relevant_snippets(rec: dict) -> list[int]:
    """Indices of the claim's most relevant snippets (by word overlap, at most 2)."""
    scored = sorted(((overlap(rec["claim"], s["text"]), i) for i, s in enumerate(rec["evidence"])),
                    reverse=True)
    return [i for score, i in scored[:MAX_SNIPPETS_PER_CLAIM] if score >= MIN_OVERLAP]


def build_weak_pairs(train: list[dict], rng: random.Random, max_per_class: int) -> list[dict]:
    """Make entail / contradict / neutral weak pairs from train claims.

    Contradict and neutral are capped at max_per_class (random, seeded); entail
    keeps everything (it is the rare class).
    """
    by_label = {"entail": [], "contradict": [], "neutral": []}
    for rec in train:
        if rec["label"] in WEAK_LABEL:
            for i in relevant_snippets(rec):
                by_label[WEAK_LABEL[rec["label"]]].append(pair(rec, i, WEAK_LABEL[rec["label"]]))
        # neutral: snippets from NEUTRAL_PER_CLAIM different, random claims
        for _ in range(NEUTRAL_PER_CLAIM):
            other = rng.choice(train)
            while other["id"] == rec["id"]:
                other = rng.choice(train)
            by_label["neutral"].append(pair(rec, rng.randrange(len(other["evidence"])), "neutral", other))
    for lab in ("contradict", "neutral"):
        rng.shuffle(by_label[lab])
        by_label[lab] = by_label[lab][:max_per_class]
    pairs = by_label["entail"] + by_label["contradict"] + by_label["neutral"]
    rng.shuffle(pairs)
    return pairs


def main() -> None:
    """Write the gold pool and the weak X-FACT pairs, and print counts."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/processed/ar/xfact_ar_train.jsonl")
    ap.add_argument("--val", default="data/processed/ar/xfact_ar_val.jsonl")
    ap.add_argument("--pool_out", default="data/nli/ar_gold_pool.jsonl")
    ap.add_argument("--pairs_out", default="data/nli/ar_train_xfact.jsonl")
    ap.add_argument("--max_per_class", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="use only N train claims (test run)")
    args = ap.parse_args()

    rng = random.Random(SEED)
    train = [r for r in read_jsonl(args.train) if r.get("evidence")]
    val = [r for r in read_jsonl(args.val) if r.get("evidence")]
    pool = pick_gold_pool(train, val, rng)
    pool_ids = {r["id"] for r in pool}
    train_rest = [r for r in train if r["id"] not in pool_ids][: args.limit]

    pairs = build_weak_pairs(train_rest, rng, args.max_per_class)
    if not args.limit:
        write_jsonl(pool, args.pool_out)
    write_jsonl(pairs, args.pairs_out if not args.limit else args.pairs_out + ".sample10")
    print(f"gold pool: {len(pool)} claims {dict(Counter(r['label'] for r in pool))}")
    print(f"train claims used: {len(train_rest)} (gold-pool claims excluded)")
    print(f"weak pairs: {len(pairs)} {dict(Counter(p['label'] for p in pairs))}")


if __name__ == "__main__":
    main()
