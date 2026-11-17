"""Build the weak NLI training pairs for fine-tuning mDeBERTa (R4, CLAUDE.md §9).

What it does
    Makes (claim, snippet, label) pairs from the English TRAIN files only (val files
    for a small monitoring set), so test claims never reach the NLI model:
        entail      supported claim    + one of its relevant snippets (relevance > 0.3)
        contradict  contradicted claim + one of its relevant snippets
                    + `_B` train twins: the claim + a snippet whose entity was swapped
        neutral     any claim + a snippet of a random other claim
    At most 2 snippets per claim for each class. The three English classes are cut to
    the same size (the smallest class, at most ``--n-per-class``), with the `_B` pairs
    kept first because they are the cleanest contradictions.
    Arabic pairs from Youssef (Y2) are added as they are: data/nli/ar_train.jsonl when
    it exists, otherwise data/nli/ar_train_xfact.jsonl.

    These labels are weak: a snippet of a false claim often repeats the claim rather
    than contradicting it. The gold set (data/nli/gold_val.jsonl) measures real quality.

Inputs
    data/processed/en/{multifc,averitec}_{train,val}.jsonl, features/relevance_en.parquet,
    data/copies/en/train.jsonl (the `_B` twins), data/nli/ar_train*.jsonl

Outputs
    data/nli/train.jsonl and data/nli/val_weak.jsonl, same fields as gold_val.jsonl:
    id, claim_id, snippet_idx, language, claim, snippet, label, origin
    results/R4/pairs_report.json  (counts per class, language and origin)

How to run (repo root; reading the parquet needs pandas + pyarrow)
    python -m src.nli.build_pairs --n-per-class 10 --out data/nli/test10/train.jsonl \
        --val-out data/nli/test10/val_weak.jsonl --report data/nli/test10/report.json
    python -m src.nli.build_pairs
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.eval.loaders import load_jsonl

EN = "data/processed/en"
TRAIN_FILES = [f"{EN}/multifc_train.jsonl", f"{EN}/averitec_train.jsonl"]
VAL_FILES = [f"{EN}/multifc_val.jsonl", f"{EN}/averitec_val.jsonl"]
TWIN_FILE = "data/copies/en/train.jsonl"
AR_FILES = ["data/nli/ar_train.jsonl", "data/nli/ar_train_xfact.jsonl"]  # first that exists
RELEVANCE_MIN = 0.3
PER_CLAIM = 2
CLASS_OF = {"supported": "entail", "contradicted": "contradict"}


def pair(claim_id: str, idx: int, claim: str, snippet: str, label: str, origin: str,
         pair_id: str | None = None) -> dict:
    """One NLI pair in the gold_val.jsonl format."""
    return {"id": pair_id or f"{claim_id}_s{idx}", "claim_id": claim_id, "snippet_idx": idx,
            "language": "en", "claim": claim, "snippet": snippet, "label": label,
            "origin": origin}


def relevant_snippets(record: dict, relevance: dict, rng: random.Random) -> list[int]:
    """Indices of up to PER_CLAIM snippets with relevance > RELEVANCE_MIN (random pick)."""
    scores = relevance.get(record["id"], {})
    good = [i for i, ev in enumerate(record["evidence"])
            if ev["text"].strip() and scores.get(i, 0.0) > RELEVANCE_MIN]
    rng.shuffle(good)
    return sorted(good[:PER_CLAIM])


def labelled_pairs(records: list[dict], relevance: dict, seed: int) -> list[dict]:
    """Entail pairs from supported claims, contradict pairs from contradicted claims."""
    out = []
    for rec in records:
        label = CLASS_OF.get(rec["label"])
        if label is None:
            continue          # unverified claims: their snippets do not settle the claim
        rng = random.Random(f"{seed}:{rec['id']}")
        for i in relevant_snippets(rec, relevance, rng):
            out.append(pair(rec["id"], i, rec["claim"], rec["evidence"][i]["text"],
                            label, f"en_{rec['label']}"))
    return out


def swap_pairs(twins: list[dict], originals: dict, seed: int) -> list[dict]:
    """Contradict pairs from `_B` twins: only snippets the swap actually changed."""
    out = []
    for twin in twins:
        if not twin["id"].endswith("_B"):
            continue
        orig = originals.get(twin["id"][:-2])
        if orig is None:
            continue
        changed = [i for i, (a, b) in enumerate(zip(orig["evidence"], twin["evidence"]))
                   if a["text"] != b["text"]]
        random.Random(f"{seed}:{twin['id']}").shuffle(changed)
        for i in sorted(changed[:PER_CLAIM]):
            out.append(pair(twin["id"], i, twin["claim"], twin["evidence"][i]["text"],
                            "contradict", "en_swap_B"))
    return out


def random_pairs(records: list[dict], n: int, seed: int) -> list[dict]:
    """``n`` neutral pairs: a claim with a snippet of a different, random claim."""
    rng = random.Random(seed)
    usable = [r for r in records if any(ev["text"].strip() for ev in r["evidence"])]
    if len(usable) < 2:
        return []
    out, seen = [], set()
    for _ in range(n * 3):            # a few extra tries for skipped duplicates
        if len(out) >= n:
            break
        rec, donor = rng.sample(usable, 2)
        if rec["claim"].strip().lower() == donor["claim"].strip().lower():
            continue                  # duplicate claim text: its snippet would be relevant
        i = rng.choice([k for k, ev in enumerate(donor["evidence"]) if ev["text"].strip()])
        pid = f"{rec['id']}_x_{donor['id']}_s{i}"
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pair(rec["id"], i, rec["claim"], donor["evidence"][i]["text"], "neutral",
                        "en_random", pair_id=pid))
    return out


def balance(entail: list[dict], contradict_swap: list[dict], contradict: list[dict],
            neutral: list[dict], n_per_class: int, seed: int) -> list[dict]:
    """Cut the three classes to the same size, keeping all `_B` pairs that fit first."""
    rng = random.Random(seed)
    n = min(n_per_class, len(entail), len(contradict_swap) + len(contradict), len(neutral))

    def take(rows: list[dict], k: int) -> list[dict]:
        rows = sorted(rows, key=lambda p: p["id"])   # fixed order before the shuffle
        rng.shuffle(rows)
        return rows[:k]

    swap = take(contradict_swap, n)
    chosen = take(entail, n) + swap + take(contradict, n - len(swap)) + take(neutral, n)
    rng.shuffle(chosen)
    return chosen


def build(records: list[dict], relevance: dict, twins: list[dict], n_per_class: int,
          seed: int = 42) -> list[dict]:
    """All balanced English pairs for one split (see the module docstring)."""
    labelled = labelled_pairs(records, relevance, seed)
    entail = [p for p in labelled if p["label"] == "entail"]
    contradict = [p for p in labelled if p["label"] == "contradict"]
    swaps = swap_pairs(twins, {r["id"]: r for r in records}, seed)
    neutral = random_pairs(records, max(len(entail), n_per_class), seed)
    return balance(entail, swaps, contradict, neutral, n_per_class, seed)


def load_relevance(path: str) -> dict:
    """features/relevance_en.parquet -> {id: {snippet_idx: relevance}}."""
    import pandas as pd

    rel: dict = {}
    for row in pd.read_parquet(path).itertuples(index=False):
        rel.setdefault(row.id, {})[int(row.snippet_idx)] = float(row.relevance)
    return rel


def count(pairs: list[dict]) -> dict:
    """Counts per label, language and origin for the report."""
    return {"n_pairs": len(pairs),
            "per_label": dict(Counter(p["label"] for p in pairs)),
            "per_language": dict(Counter(p["language"] for p in pairs)),
            "per_origin": dict(Counter(p["origin"] for p in pairs))}


def write_jsonl(rows: list[dict], path: str) -> None:
    """Write rows as UTF-8 JSON Lines (Arabic kept readable)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """CLI: build train + val pairs and the report."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-files", nargs="+", default=TRAIN_FILES)
    ap.add_argument("--val-files", nargs="+", default=VAL_FILES)
    ap.add_argument("--twins", default=TWIN_FILE)
    ap.add_argument("--relevance", default="features/relevance_en.parquet")
    ap.add_argument("--ar-files", nargs="+", default=AR_FILES)
    ap.add_argument("--n-per-class", type=int, default=10_000)
    ap.add_argument("--n-val-per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/nli/train.jsonl")
    ap.add_argument("--val-out", default="data/nli/val_weak.jsonl")
    ap.add_argument("--report", default="results/R4/pairs_report.json")
    args = ap.parse_args()

    relevance = load_relevance(args.relevance)
    train_records = [r for p in args.train_files for r in load_jsonl(p)]
    val_records = [r for p in args.val_files for r in load_jsonl(p)]
    twins = load_jsonl(args.twins) if Path(args.twins).exists() else []
    train = build(train_records, relevance, twins, args.n_per_class, args.seed)
    val = build(val_records, relevance, [], args.n_val_per_class, args.seed)

    ar_file = next((p for p in args.ar_files if Path(p).exists()), None)
    arabic = load_jsonl(ar_file) if ar_file else []
    if args.n_per_class < 100:        # 10-row check: keep the Arabic part small too
        arabic = arabic[:args.n_per_class]
    train = train + arabic
    random.Random(args.seed).shuffle(train)

    write_jsonl(train, args.out)
    write_jsonl(val, args.val_out)
    report = {"train": count(train), "val_weak": count(val), "arabic_file": ar_file,
              "seed": args.seed, "relevance_min": RELEVANCE_MIN, "per_claim": PER_CLAIM}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
