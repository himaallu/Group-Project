"""Build the weak NLI training pairs for fine-tuning mDeBERTa (R4, CLAUDE.md §9).

What it does
    Makes (claim, snippet, label) pairs from the TRAIN files only (val files for a
    small monitoring set), so test claims never reach the NLI model.

    v1 (no ``--filter-model``): labels come straight from the claim label.
        entail      supported claim    + one of its relevant snippets (relevance > 0.3)
        contradict  contradicted claim + one of its relevant snippets
                    + `_B` train twins: the claim + a snippet whose entity was swapped
        neutral     any claim + a snippet of a random other claim
    v1 result: gold macro-F1 fell (EN 0.37 -> 0.12). Most gold pairs are a claim's OWN
    snippet that does not settle it (neutral), and v1 had no such examples, so the
    model learned "on-topic snippet -> copy the claim's label".

    v2 (``--filter-model <zero-shot model>``) fixes this in two ways:
        1. Agreement filter: a claim-label pair (entail / contradict) is kept only if
           the zero-shot model also gives that label as its top class (prob >= 0.5).
        2. Hard neutrals: a claim + one of its own snippets, kept when the zero-shot
           model calls it neutral (prob >= 0.5). Neutral = these + random-snippet pairs.
        `_B` swap pairs are kept without the filter: they contradict by construction.
        Youssef's Arabic pairs get the same filter, plus Arabic hard neutrals from
        X-FACT Arabic train claims; the Arabic part is balanced like the English one.

    At most 2 snippets per claim per class in v1 (4 candidates per claim in v2, since
    the filter drops many). Classes are cut to the same size (the smallest class, at
    most ``--n-per-class``), with the `_B` pairs kept first.

Inputs
    data/processed/en/{multifc,averitec}_{train,val}.jsonl, features/relevance_en.parquet,
    data/copies/en/train.jsonl (the `_B` twins), data/nli/ar_train*.jsonl,
    data/processed/ar/xfact_ar_train.jsonl (v2 Arabic hard neutrals)

Outputs
    data/nli/train.jsonl and data/nli/val_weak.jsonl, same fields as gold_val.jsonl:
    id, claim_id, snippet_idx, language, claim, snippet, label, origin
    results/R4/pairs_report.json  (counts per class, language and origin)

How to run (repo root; reading the parquet needs pandas + pyarrow; v2 needs torch)
    python -m src.nli.build_pairs --n-per-class 10 --out data/nli/test10/train.jsonl \
        --val-out data/nli/test10/val_weak.jsonl --report data/nli/test10/report.json
    python -m src.nli.build_pairs                                     # v1
    python -m src.nli.build_pairs --filter-model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable

from src.eval.loaders import load_jsonl

EN = "data/processed/en"
TRAIN_FILES = [f"{EN}/multifc_train.jsonl", f"{EN}/averitec_train.jsonl"]
VAL_FILES = [f"{EN}/multifc_val.jsonl", f"{EN}/averitec_val.jsonl"]
TWIN_FILE = "data/copies/en/train.jsonl"
AR_FILES = ["data/nli/ar_train.jsonl", "data/nli/ar_train_xfact.jsonl"]  # first that exists
AR_RECORDS = "data/processed/ar/xfact_ar_train.jsonl"
RELEVANCE_MIN = 0.3
PER_CLAIM = 2
CANDIDATES_PER_CLAIM = 4    # v2: more candidates per claim, because the filter drops many
MIN_PROB = 0.5              # v2: the zero-shot top class must have at least this probability
CLASS_OF = {"supported": "entail", "contradicted": "contradict"}

# scorer(claims, snippets) -> [{"p_entail", "p_contradict", "p_neutral"}, ...]
Scorer = Callable[[list[str], list[str]], list[dict]]


def pair(claim_id: str, idx: int, claim: str, snippet: str, label: str, origin: str,
         pair_id: str | None = None, language: str = "en") -> dict:
    """One NLI pair in the gold_val.jsonl format."""
    return {"id": pair_id or f"{claim_id}_s{idx}", "claim_id": claim_id, "snippet_idx": idx,
            "language": language, "claim": claim, "snippet": snippet, "label": label,
            "origin": origin}


def relevant_snippets(record: dict, relevance: dict | None, rng: random.Random,
                      per_claim: int = PER_CLAIM) -> list[int]:
    """Indices of up to ``per_claim`` non-empty snippets (random pick).

    With ``relevance`` only snippets scoring > RELEVANCE_MIN count; with None, any.
    """
    scores = relevance.get(record["id"], {}) if relevance is not None else None
    good = [i for i, ev in enumerate(record["evidence"])
            if ev["text"].strip() and (scores is None or scores.get(i, 0.0) > RELEVANCE_MIN)]
    rng.shuffle(good)
    return sorted(good[:per_claim])


def labelled_pairs(records: list[dict], relevance: dict, seed: int,
                   per_claim: int = PER_CLAIM) -> list[dict]:
    """Entail pairs from supported claims, contradict pairs from contradicted claims."""
    out = []
    for rec in records:
        label = CLASS_OF.get(rec["label"])
        if label is None:
            continue          # unverified claims: their snippets do not settle the claim
        rng = random.Random(f"{seed}:{rec['id']}")
        for i in relevant_snippets(rec, relevance, rng, per_claim):
            out.append(pair(rec["id"], i, rec["claim"], rec["evidence"][i]["text"],
                            label, f"en_{rec['label']}"))
    return out


def own_snippet_pairs(records: list[dict], seed: int, per_claim: int = CANDIDATES_PER_CLAIM,
                      language: str = "en") -> list[dict]:
    """v2 hard-neutral candidates: a claim + one of its OWN snippets, any claim label.

    These look like the gold pairs (same topic, often not deciding the claim). Only
    the ones the zero-shot model calls neutral are kept (see ``filter_by_model``).
    """
    out = []
    for rec in records:
        rng = random.Random(f"{seed}:own:{rec['id']}")
        for i in relevant_snippets(rec, None, rng, per_claim):
            out.append(pair(rec["id"], i, rec["claim"], rec["evidence"][i]["text"], "neutral",
                            f"{language}_own_neutral", language=language))
    return out


def filter_by_model(pairs: list[dict], scorer: Scorer, min_prob: float = MIN_PROB) -> list[dict]:
    """Keep the pairs whose label is also the scorer's top class with prob >= min_prob.

    Each distinct (claim, snippet) is scored once, even if it is a candidate for two
    labels (e.g. entail and hard neutral); at most one of them can then survive.
    """
    keys = sorted({(p["claim"], p["snippet"]) for p in pairs})
    if not keys:
        return []
    probs = dict(zip(keys, scorer([k[0] for k in keys], [k[1] for k in keys])))
    kept = []
    for p in pairs:
        pr = probs[(p["claim"], p["snippet"])]
        top = max(pr, key=pr.get)                         # e.g. "p_entail"
        if top == f"p_{p['label']}" and pr[top] >= min_prob:
            kept.append(p)
    return kept


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


def take(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    """``k`` rows picked at random, reproducibly (fixed order before the shuffle)."""
    rows = sorted(rows, key=lambda p: p["id"])
    rng.shuffle(rows)
    return rows[:k]


def balance(entail: list[dict], contradict_swap: list[dict], contradict: list[dict],
            neutral: list[dict], n_per_class: int, seed: int) -> list[dict]:
    """Cut the three classes to the same size, keeping all `_B` pairs that fit first."""
    rng = random.Random(seed)
    n = min(n_per_class, len(entail), len(contradict_swap) + len(contradict), len(neutral))
    swap = take(contradict_swap, n, rng)
    chosen = (take(entail, n, rng) + swap + take(contradict, n - len(swap), rng)
              + take(neutral, n, rng))
    rng.shuffle(chosen)
    return chosen


def build(records: list[dict], relevance: dict, twins: list[dict], n_per_class: int,
          seed: int = 42, scorer: Scorer | None = None) -> list[dict]:
    """All balanced English pairs for one split (v1, or v2 when ``scorer`` is given)."""
    swaps = swap_pairs(twins, {r["id"]: r for r in records}, seed)
    if scorer is None:                                      # v1
        labelled = labelled_pairs(records, relevance, seed)
        hard = []
    else:                                                   # v2: filter + hard neutrals
        candidates = (labelled_pairs(records, relevance, seed, CANDIDATES_PER_CLAIM)
                      + own_snippet_pairs(records, seed))
        kept = filter_by_model(candidates, scorer)
        labelled = [p for p in kept if p["label"] != "neutral"]
        hard = [p for p in kept if p["label"] == "neutral"]
    entail = [p for p in labelled if p["label"] == "entail"]
    contradict = [p for p in labelled if p["label"] == "contradict"]
    # v2: about as many random neutrals as hard ones, so neutral is roughly half each
    neutral = hard + random_pairs(records, max(len(hard), len(entail), n_per_class), seed)
    return balance(entail, swaps, contradict, neutral, n_per_class, seed)


def build_arabic(ar_pairs: list[dict], ar_records: list[dict], scorer: Scorer,
                 seed: int = 42) -> list[dict]:
    """v2 Arabic part: filtered Y2 pairs + Arabic hard neutrals, balanced per class."""
    kept = filter_by_model([p for p in ar_pairs if p["label"] != "neutral"], scorer)
    hard = filter_by_model(own_snippet_pairs(ar_records, seed, PER_CLAIM, "ar"), scorer)
    entail = [p for p in kept if p["label"] == "entail"]
    contradict = [p for p in kept if p["label"] == "contradict"]
    neutral = hard + [p for p in ar_pairs if p["label"] == "neutral"]
    return balance(entail, [], contradict, neutral, len(ar_pairs), seed)


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
    ap.add_argument("--ar-records", default=AR_RECORDS)
    ap.add_argument("--filter-model", help="zero-shot NLI model for the v2 filter (needs torch)")
    ap.add_argument("--n-per-class", type=int, default=10_000)
    ap.add_argument("--n-val-per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/nli/train.jsonl")
    ap.add_argument("--val-out", default="data/nli/val_weak.jsonl")
    ap.add_argument("--report", default="results/R4/pairs_report.json")
    args = ap.parse_args()

    scorer = None
    small = args.n_per_class < 100     # 10-row check: keep every part small
    if args.filter_model:
        from src.nli.score import score_pairs

        def scorer(claims: list[str], snippets: list[str]) -> list[dict]:
            """Zero-shot NLI probabilities from the filter model."""
            return score_pairs(claims, snippets, model=args.filter_model)

    relevance = load_relevance(args.relevance)
    train_records = [r for p in args.train_files for r in load_jsonl(p)]
    val_records = [r for p in args.val_files for r in load_jsonl(p)]
    if small:
        train_records, val_records = train_records[:200], val_records[:200]
    twins = load_jsonl(args.twins) if Path(args.twins).exists() else []
    train = build(train_records, relevance, twins, args.n_per_class, args.seed, scorer)
    val = build(val_records, relevance, [], args.n_val_per_class, args.seed, scorer)

    ar_file = next((p for p in args.ar_files if Path(p).exists()), None)
    arabic = load_jsonl(ar_file) if ar_file else []
    if small:
        arabic = arabic[:args.n_per_class]
    if scorer is not None and arabic:
        ar_records = load_jsonl(args.ar_records) if Path(args.ar_records).exists() else []
        arabic = build_arabic(arabic, ar_records[:200] if small else ar_records, scorer,
                              args.seed)
    train = train + arabic
    random.Random(args.seed).shuffle(train)

    write_jsonl(train, args.out)
    write_jsonl(val, args.val_out)
    report = {"version": "v2" if scorer else "v1", "filter_model": args.filter_model,
              "min_prob": MIN_PROB if scorer else None,
              "train": count(train), "val_weak": count(val), "arabic_file": ar_file,
              "seed": args.seed, "relevance_min": RELEVANCE_MIN, "per_claim": PER_CLAIM}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
