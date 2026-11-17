"""Unit tests for the R1 data code (src/data/common.py, src/data/clean.py).

What it does
    Builds a 10-row toy dataset in the shared format and checks each cleaning rule,
    the stratified split, the label map and the URL-domain helper.

How to run (from the repo root)
    python -m pytest tests/data -q
"""

from src.data.clean import claim_overlap, clean_dataset, is_factcheck_site
from src.data.common import (domain_of, load_label_map, make_evidence, make_record,
                             map_label, stratified_split)


def ev(text, url="https://news.example.com/a"):
    """Short helper: one evidence snippet."""
    return make_evidence(text, url)


def toy_records():
    """10 records covering every rule (ids toy_000000 .. toy_000009)."""
    filler = [ev(f"Background fact number {i} about the city council budget.") for i in range(12)]
    rows = [
        # 0: one fact-check snippet (rule 1), one fact-check path (rule 1), one good
        ("Dubai Metro suspended all lines today", "contradicted",
         [ev("Snopes rates this false.", "https://www.snopes.com/fact-check/metro/"),
          ev("Reuters checks the metro claim.", "https://www.reuters.com/article/factcheck-metro"),
          ev("Green Line operating normally, RTA confirms.")]),
        # 1: snippet that repeats the claim (rule 2) + one good
        ("The mayor opened a new bridge in Ajman", "supported",
         [ev("Breaking: the mayor opened a new bridge in Ajman yesterday."),
          ev("Officials attended a ribbon cutting on Sunday.")]),
        # 2: 12 snippets -> capped at 10 (rule 3)
        ("City council approved the budget", "supported", filler),
        # 3: exact duplicate of claim 2 with other casing/punctuation (rule 3)
        ("City Council approved the budget!", "supported", [ev("Budget passed 7-2.")]),
        # 4: only fact-check snippets -> no snippets left (rule 4)
        ("A shark was seen in the mall fountain", "contradicted",
         [ev("Misbar: fake.", "https://misbar.com/factcheck/1"),
          ev("AFP Factuel: faux.", "https://factuel.afp.com/x")]),
        # 5-9: plain records
        ("Rain is expected in Abu Dhabi on Friday", "supported", [ev("Forecast shows showers Friday.")]),
        ("The airport closed for a week", "contradicted", [ev("Flights ran on schedule all week.")]),
        ("A new law bans all cars downtown", "unverified", [ev("No official statement was found.")]),
        ("The museum is free on Mondays", "supported", [ev("Entry is free every Monday.")]),
        ("Schools will start in July", "contradicted", [ev("The term begins in late August.")]),
    ]
    return [make_record(f"toy_{i:06d}", "toy", "train", claim, evidence, label, label)
            for i, (claim, label, evidence) in enumerate(rows)]


def test_rule1_factcheck_domain_and_path():
    """Fact-check domains (with subdomains) and fact-check URL paths are detected."""
    domains = {"snopes.com", "afp.com"}
    assert is_factcheck_site(ev("x", "https://www.snopes.com/a"), domains)
    assert is_factcheck_site(ev("x", "https://factuel.afp.com/a"), domains)
    assert is_factcheck_site(ev("x", "https://www.reuters.com/article/factcheck-x"), domains)
    assert not is_factcheck_site(ev("x", "https://gulfnews.com/uae/metro"), domains)


def test_rule2_overlap():
    """Overlap is the share of claim words found in the snippet."""
    assert claim_overlap("the mayor opened a bridge", "The mayor opened a bridge today") == 1.0
    assert claim_overlap("the mayor opened a bridge", "nothing related") == 0.0


def test_clean_dataset_counts_each_rule():
    """Every rule removes what it should, and the report counts it."""
    splits, no_snippets, report = clean_dataset(toy_records(), split_mode="keep")
    counts = report["counts"]
    assert counts["claims_in"] == 10
    assert counts["rule1_factcheck_site"] == 4      # 2 in record 0, 2 in record 4
    assert counts["rule2_overlap_gt_90"] == 1       # record 1
    assert counts["rule3_over_10_snippets"] == 2    # record 2: 12 -> 10
    assert counts["rule3_duplicate_claims"] == 1    # record 3
    assert counts["rule4_claims_without_snippets"] == 1
    assert [r["id"] for r in no_snippets] == ["toy_000004"]

    kept = [r for part in splits.values() for r in part]
    assert len(kept) == 8                           # 10 - duplicate - no snippets
    assert all(len(r["evidence"]) <= 10 for r in kept)
    record0 = next(r for r in kept if r["id"] == "toy_000000")
    assert [e["domain"] for e in record0["evidence"]] == ["news.example.com"]


def test_keep_mode_carves_val_from_train():
    """--split keep moves about 10% of train to val when there is no val split."""
    records = toy_records()  # all "train"
    for i, rec in enumerate(records):
        rec["claim"] = f"{rec['claim']} variant {i}"  # avoid duplicate removal
    splits, _, _ = clean_dataset(records, split_mode="keep")
    assert len(splits["val"]) >= 1 and len(splits["test"]) == 0


def test_duplicate_kept_in_test_not_train():
    """When a claim is in train and test, the test copy survives."""
    original = toy_records()[0]
    dup = dict(original, id="toy_000099", split="test")
    splits, _, report = clean_dataset([original, dup], split_mode="keep")
    assert [r["id"] for r in splits["test"]] == ["toy_000099"]
    assert splits["train"] == []
    assert report["counts"]["rule3_duplicate_claims"] == 1


def test_stratified_split_is_reproducible_and_balanced():
    """Same seed -> same split; each label is split about 80/10/10."""
    records = [{"id": str(i), "label": "supported" if i % 2 else "contradicted"}
               for i in range(100)]
    a = stratified_split(records, seed=42)
    b = stratified_split(records, seed=42)
    assert [r["id"] for r in a["test"]] == [r["id"] for r in b["test"]]
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (80, 10, 10)
    assert sum(r["label"] == "supported" for r in a["test"]) == 5


def test_label_map_rules():
    """Key labels from CLAUDE.md section 4 map as agreed."""
    label_map = load_label_map()
    assert map_label(label_map, "multifc", "Pants on Fire!") == "contradicted"
    assert map_label(label_map, "multifc", "mostly true") == "supported"
    assert map_label(label_map, "multifc", "unproven") == "unverified"
    assert map_label(label_map, "multifc", "half-true") == "dropped"
    assert map_label(label_map, "averitec", "Refuted") == "contradicted"
    assert map_label(label_map, "multifc", "a label nobody uses") is None


def test_domain_of_unwraps_archive_links():
    """domain_of drops www. and reads the original site inside web.archive.org links."""
    assert domain_of("https://www.gulfnews.com/uae/x") == "gulfnews.com"
    assert domain_of("https://web.archive.org/web/20201129/https://scoopertino.com/a") == "scoopertino.com"
    assert domain_of("") == ""
