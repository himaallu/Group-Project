"""Clean and split any TwinCheck dataset in the shared format (R1, CLAUDE.md section 4).

What it does (the five cleaning rules, in this order)
    1. Drop snippets from fact-checking sites (domain list below, subdomains
       included) and snippets whose URL path is a fact-check page ("fact-check",
       "factcheck", "fact-checker" in the path, e.g. reuters.com/article/factcheck-...).
       These pages state the verdict, so a model could copy it instead of reasoning.
    2. Drop snippets that repeat the claim: more than 90% of the claim's words
       appear in the snippet.
    3. Remove duplicate claims (same words, ignoring case and punctuation); remove
       repeated snippets inside a claim; keep at most 10 snippets per claim.
       When duplicates sit in different splits, the copy in test (then val) is
       kept so the same claim never appears in train and test.
    4. Claims left with no snippets go to a separate file and are counted.
    5. Split. --split own: our own 80/10/10 train/val/test split, stratified by
       label, seed 42 (MultiFC, whose test labels were never released).
       --split keep: keep the dataset's split; if it has no val records, 10% of
       train becomes val (stratified, seed 42).

Inputs
    --input        shared-format JSONL (e.g. data/processed/en/staging/multifc_all.jsonl)
    --name         output name prefix, e.g. multifc
    --out-dir      e.g. data/processed/en
    --split        own | keep
    --extra-domains  optional text file, one extra fact-check domain per line

Outputs
    <out-dir>/<name>_{train,val,test}.jsonl   cleaned records
    <out-dir>/<name>_nosnippets.jsonl         claims with no snippets left (rule 4)
    --report (default results/R1/clean_report_<name>.json): counts per rule,
        per split and label, and the domains removed most by rule 1

How to run (from the repo root)
    python -m src.data.clean --input data/processed/en/staging/multifc_all.jsonl \
        --name multifc --out-dir data/processed/en --split own
    # Youssef (Arabic):
    python -m src.data.clean --input <xfact_ar_all.jsonl> --name xfact_ar \
        --out-dir data/processed/ar --split keep --report results/Y1/clean_report_xfact_ar.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from urllib.parse import urlparse

from src.data.common import LABELS, SEED, SPLITS, read_jsonl, stratified_split, write_jsonl

MAX_SNIPPETS = 10
OVERLAP_THRESHOLD = 0.9  # rule 2: drop if MORE than 90% of claim words are in the snippet

# Rule 1. A snippet is dropped if its domain is one of these or a subdomain of one.
FACTCHECK_DOMAINS = {
    # English fact-checkers (incl. the MultiFC source sites)
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org",
    "truthorfiction.com", "hoax-slayer.net", "hoax-slayer.com", "leadstories.com",
    "africacheck.org", "checkyourfact.com", "factly.in", "boomlive.in", "altnews.in",
    "factscan.ca", "climatefeedback.org", "sciencefeedback.co", "healthfeedback.org",
    "gossipcop.com", "factcheckni.org", "pesacheck.org", "theferret.scot",
    "verafiles.org", "factchecker.in", "polygraph.info", "mediawise.org",
    "teyit.org", "maldita.es", "newtral.es", "correctiv.org",
    # Arabic fact-checkers (from Youssef, notes/Y1_for_rathi/arabic_factcheck_domains.txt)
    "misbar.com", "falsoo.com", "afp.com", "fatabyyano.net", "verify-sy.com",
    "dabegad.com", "akeed.jo", "icheck.tn", "matsda2sh.com", "kashif.ps",
}
# Rule 1, second part: fact-check pages on general news sites (Reuters, AP, USA Today,
# Washington Post Fact Checker, ABC/RMIT Fact Check, ...) are found by their URL path.
FACTCHECK_PATH = re.compile(r"fact[-_]?check", re.IGNORECASE)

WORD = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> list[str]:
    """Lower-case word tokens (works for English and Arabic)."""
    return WORD.findall(text.lower())


def is_factcheck_site(snippet: dict, domains: set[str]) -> bool:
    """Rule 1: True if the snippet comes from a fact-checking site or page.

    Args:
        snippet: evidence dict with ``url`` and ``domain``.
        domains: set of fact-check domains.

    Returns:
        True when the domain (or a parent domain) is listed, or the URL path
        looks like a fact-check page.
    """
    domain = snippet.get("domain", "")
    parts = domain.split(".")
    # "factuel.afp.com" -> check "factuel.afp.com", "afp.com", "com"
    if any(".".join(parts[i:]) in domains for i in range(len(parts))):
        return True
    path = urlparse(snippet.get("url", "")).path
    return bool(FACTCHECK_PATH.search(path))


def claim_overlap(claim: str, snippet_text: str) -> float:
    """Rule 2: share of the claim's distinct words that also appear in the snippet.

    Args:
        claim: claim text.
        snippet_text: snippet text.

    Returns:
        Number in [0, 1]; 0 when the claim has no words.
    """
    claim_words = set(words(claim))
    if not claim_words:
        return 0.0
    return len(claim_words & set(words(snippet_text))) / len(claim_words)


def claim_key(claim: str) -> str:
    """Rule 3: normalised claim text used to find duplicates."""
    return " ".join(words(claim))


def clean_snippets(record: dict, domains: set[str], stats: Counter,
                   removed_domains: Counter) -> list[dict]:
    """Apply rules 1, 2 and the snippet part of rule 3 to one record.

    Args:
        record: shared-format record.
        domains: fact-check domains for rule 1.
        stats: counter updated with how many snippets each rule removed.
        removed_domains: counter of domains removed by rule 1.

    Returns:
        The cleaned evidence list (at most ``MAX_SNIPPETS`` items).
    """
    kept, seen_texts = [], set()
    for snippet in record["evidence"]:
        stats["snippets_in"] += 1
        if is_factcheck_site(snippet, domains):
            stats["rule1_factcheck_site"] += 1
            removed_domains[snippet.get("domain", "")] += 1
            continue
        if claim_overlap(record["claim"], snippet["text"]) > OVERLAP_THRESHOLD:
            stats["rule2_overlap_gt_90"] += 1
            continue
        text_key = " ".join(words(snippet["text"]))
        if not text_key or text_key in seen_texts:
            stats["rule3_duplicate_or_empty_snippet"] += 1
            continue
        seen_texts.add(text_key)
        kept.append(snippet)
    if len(kept) > MAX_SNIPPETS:
        stats["rule3_over_10_snippets"] += len(kept) - MAX_SNIPPETS
        kept = kept[:MAX_SNIPPETS]  # loaders keep the source's rank order
    stats["snippets_out"] += len(kept)
    return kept


def remove_duplicate_claims(records: list[dict], stats: Counter) -> list[dict]:
    """Rule 3: keep one record per claim text.

    Records are visited test first, then val, then train, so when a claim appears
    in two splits the test (or val) copy is kept and train never sees it.

    Args:
        records: shared-format records.
        stats: counter updated with the number of duplicates removed.

    Returns:
        Records without duplicate claims, in their original order.
    """
    priority = {"test": 0, "val": 1, "train": 2}
    order = sorted(range(len(records)), key=lambda i: priority.get(records[i]["split"], 3))
    seen, keep = {}, set()
    for i in order:
        key = claim_key(records[i]["claim"])
        if key in seen:
            stats["rule3_duplicate_claims"] += 1
            if records[seen[key]]["label"] != records[i]["label"]:
                stats["rule3_duplicate_claims_with_other_label"] += 1
            continue
        seen[key] = i
        keep.add(i)
    return [rec for i, rec in enumerate(records) if i in keep]


def assign_splits(records: list[dict], mode: str, seed: int = SEED) -> list[dict]:
    """Rule 5: set the ``split`` field of every record.

    Args:
        records: shared-format records.
        mode: ``"own"`` (80/10/10 stratified) or ``"keep"`` (dataset split; carve
            10% of train as val when there is no val).
        seed: random seed.

    Returns:
        The same records with ``split`` set.
    """
    if mode == "own":
        parts = stratified_split(records, (0.8, 0.1, 0.1), SPLITS, seed)
    elif mode == "keep":
        has_val = any(r["split"] == "val" for r in records)
        train = [r for r in records if r["split"] == "train"]
        parts = {s: [r for r in records if r["split"] == s] for s in SPLITS}
        if not has_val and train:
            carved = stratified_split(train, (0.9, 0.1), ("train", "val"), seed)
            parts["train"], parts["val"] = carved["train"], carved["val"]
    else:
        raise ValueError(f"unknown split mode {mode!r}")
    for split, part in parts.items():
        for rec in part:
            rec["split"] = split
    return records


def read_domain_file(path: str | None) -> set[str]:
    """Read extra fact-check domains (one per line, '#' starts a comment)."""
    if not path:
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.split("#")[0].strip().lower() for line in f
                if line.split("#")[0].strip()}


def clean_dataset(records: list[dict], split_mode: str, extra_domains: set[str] = frozenset(),
                  seed: int = SEED) -> tuple[dict[str, list[dict]], list[dict], dict]:
    """Run all five rules on one dataset.

    Args:
        records: shared-format records (any language, any source).
        split_mode: ``"own"`` or ``"keep"``.
        extra_domains: more fact-check domains for rule 1.
        seed: random seed for the split.

    Returns:
        (splits, no_snippets, report): ``splits`` maps train/val/test to records
        with snippets; ``no_snippets`` holds rule-4 claims; ``report`` has counts.
    """
    domains = FACTCHECK_DOMAINS | set(extra_domains)
    stats, removed_domains = Counter(), Counter()
    stats["claims_in"] = len(records)

    for rec in records:  # rules 1-3 on snippets
        rec["evidence"] = clean_snippets(rec, domains, stats, removed_domains)
    records = remove_duplicate_claims(records, stats)  # rule 3 on claims
    records = assign_splits(records, split_mode, seed)  # rule 5 (before 4, so
    # the no-snippet file also carries a valid, reproducible split)

    no_snippets = [r for r in records if not r["evidence"]]  # rule 4
    with_snippets = [r for r in records if r["evidence"]]
    stats["rule4_claims_without_snippets"] = len(no_snippets)
    splits = {s: [r for r in with_snippets if r["split"] == s] for s in SPLITS}

    report = {
        "split_mode": split_mode,
        "seed": seed,
        "counts": dict(stats),
        "per_split_label": {s: {lab: sum(r["label"] == lab for r in part) for lab in LABELS}
                            for s, part in splits.items()},
        "no_snippets_per_label": {lab: sum(r["label"] == lab for r in no_snippets)
                                  for lab in LABELS},
        "rule1_top_domains": dict(removed_domains.most_common(25)),
    }
    return splits, no_snippets, report


def main() -> None:
    """Parse arguments, clean one dataset, write the split files and the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=["own", "keep"], required=True)
    parser.add_argument("--extra-domains", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    splits, no_snippets, report = clean_dataset(
        records, args.split, read_domain_file(args.extra_domains), args.seed)

    for split, part in splits.items():
        write_jsonl(part, os.path.join(args.out_dir, f"{args.name}_{split}.jsonl"))
    write_jsonl(no_snippets, os.path.join(args.out_dir, f"{args.name}_nosnippets.jsonl"))

    report_path = args.report or os.path.join("results", "R1", f"clean_report_{args.name}.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: report[k] for k in ("counts", "per_split_label")}, indent=2))
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
