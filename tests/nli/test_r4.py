"""Unit tests for R4 (weak NLI pairs and the gold-set report) on 10 toy rows.

No GPU, torch or model download needed.
Run: python -m pytest tests/nli/test_r4.py -q
"""

from collections import Counter

from src.nli.build_pairs import build, labelled_pairs, random_pairs, swap_pairs
from src.nli.train import class_report


def toy_records(n: int = 10) -> list[dict]:
    """n claims (even = supported, odd = contradicted), 3 snippets each."""
    return [{"id": f"multifc_{i:06d}", "claim": f"claim {i} about Dubai",
             "label": "supported" if i % 2 == 0 else "contradicted",
             "evidence": [{"text": f"snippet {j} of claim {i} in Dubai", "url": "", "domain": ""}
                          for j in range(3)]} for i in range(n)]


RECORDS = toy_records()
# snippet 0 and 1 relevant, snippet 2 not
RELEVANCE = {r["id"]: {0: 0.9, 1: 0.8, 2: 0.1} for r in RECORDS}


def swap_twin(record: dict) -> dict:
    """A `_B` twin with "Dubai" -> "Paris" in every snippet."""
    return {**record, "id": record["id"] + "_B", "label": "contradicted",
            "evidence": [{**ev, "text": ev["text"].replace("Dubai", "Paris")}
                         for ev in record["evidence"]]}


def test_labelled_pairs_use_only_relevant_snippets():
    pairs = labelled_pairs(RECORDS, RELEVANCE, seed=42)
    assert len(pairs) == 20                               # 2 relevant snippets x 10 claims
    assert all(p["snippet_idx"] in (0, 1) for p in pairs)
    for p in pairs:
        expected = "entail" if int(p["claim_id"][-1]) % 2 == 0 else "contradict"
        assert p["label"] == expected
        assert p["snippet"] == RECORDS[int(p["claim_id"][-1])]["evidence"][p["snippet_idx"]]["text"]


def test_unverified_claims_give_no_pairs():
    rec = dict(RECORDS[0], label="unverified")
    assert labelled_pairs([rec], RELEVANCE, 42) == []


def test_swap_pairs_only_changed_snippets():
    twin = swap_twin(RECORDS[0])
    twin["evidence"][1]["text"] = RECORDS[0]["evidence"][1]["text"]   # snippet 1 unchanged
    pairs = swap_pairs([twin], {r["id"]: r for r in RECORDS}, 42)
    assert {p["snippet_idx"] for p in pairs} <= {0, 2} and len(pairs) == 2
    assert all(p["label"] == "contradict" and "Paris" in p["snippet"] for p in pairs)
    assert all(p["claim_id"] == "multifc_000000_B" for p in pairs)


def test_random_pairs_never_use_own_snippet():
    pairs = random_pairs(RECORDS, 15, 42)
    assert len(pairs) == 15 and len({p["id"] for p in pairs}) == 15
    for p in pairs:
        assert f"of claim {int(p['claim_id'][-1])} " not in p["snippet"]
        assert p["label"] == "neutral"


def test_build_is_balanced_and_reproducible():
    twins = [swap_twin(r) for r in RECORDS[:2]]
    pairs = build(RECORDS, RELEVANCE, twins, n_per_class=6, seed=42)
    assert Counter(p["label"] for p in pairs) == {"entail": 6, "contradict": 6, "neutral": 6}
    assert sum(p["origin"] == "en_swap_B" for p in pairs) == 4   # all swap pairs kept first
    assert pairs == build(RECORDS, RELEVANCE, twins, n_per_class=6, seed=42)
    # the smallest class sets the size: 10 entail pairs available
    big = build(RECORDS, RELEVANCE, [], n_per_class=1000, seed=42)
    assert Counter(p["label"] for p in big) == {"entail": 10, "contradict": 10, "neutral": 10}


def test_class_report():
    gold = ["entail", "entail", "contradict", "neutral", "neutral"]
    pred = ["entail", "neutral", "contradict", "neutral", "entail"]
    rep = class_report(gold, pred)
    assert rep["accuracy"] == 0.6
    assert rep["per_class"]["contradict"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                                              "support": 1}
    assert rep["per_class"]["entail"]["f1"] == 0.5
    assert rep["macro_f1"] == round((0.5 + 1.0 + 0.5) / 3, 4)
    assert rep["confusion (rows = gold)"]["entail"]["neutral"] == 1
    # a class absent from gold and predictions is left out of the macro average
    assert class_report(["neutral", "contradict"], ["neutral", "contradict"])["macro_f1"] == 1.0
