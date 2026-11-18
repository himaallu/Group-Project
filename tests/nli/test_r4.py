"""Unit tests for R4 (weak NLI pairs and the gold-set report) on 10 toy rows.

No GPU, torch or model download needed.
Run: python -m pytest tests/nli/test_r4.py -q
"""

from collections import Counter

from src.nli.build_pairs import (build, build_arabic, filter_by_model, labelled_pairs,
                                  own_snippet_pairs, random_pairs, swap_pairs)
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


def fake_scorer(claims: list[str], snippets: list[str]) -> list[dict]:
    """Toy zero-shot model: snippet 0 entails, 1 is neutral, 2 contradicts, others unsure."""
    def probs(snippet: str) -> dict:
        if "snippet 0 " in snippet:
            return {"p_entail": 0.8, "p_contradict": 0.1, "p_neutral": 0.1}
        if "snippet 1 " in snippet:
            return {"p_entail": 0.1, "p_contradict": 0.1, "p_neutral": 0.8}
        if "snippet 2 " in snippet:
            return {"p_entail": 0.1, "p_contradict": 0.7, "p_neutral": 0.2}
        return {"p_entail": 0.2, "p_contradict": 0.45, "p_neutral": 0.35}   # top < 0.5
    return [probs(s) for s in snippets]


def test_filter_keeps_only_agreeing_confident_pairs():
    own = own_snippet_pairs(RECORDS[:2], seed=42)
    assert len(own) == 6 and all(p["label"] == "neutral" for p in own)
    kept = filter_by_model(labelled_pairs(RECORDS, RELEVANCE, 42) + own, fake_scorer)
    # entail only from supported claims' snippet 0; contradicted claims only use their
    # relevant snippets 0/1, which the model does not call contradict; hard neutrals = snippet 1
    assert {(p["label"], p["snippet_idx"]) for p in kept} == {("entail", 0), ("neutral", 1)}
    assert all(p["claim_id"] in ("multifc_000000", "multifc_000001")
               for p in kept if p["label"] == "neutral")


def test_build_v2_has_hard_neutrals_and_keeps_swaps():
    twins = [swap_twin(r) for r in RECORDS[:2]]
    pairs = build(RECORDS, RELEVANCE, twins, n_per_class=4, seed=42, scorer=fake_scorer)
    assert Counter(p["label"] for p in pairs) == {"entail": 4, "contradict": 4, "neutral": 4}
    assert all(p["origin"] == "en_swap_B" for p in pairs if p["label"] == "contradict")
    origins = {p["origin"] for p in pairs if p["label"] == "neutral"}
    assert origins <= {"en_own_neutral", "en_random"}
    assert pairs == build(RECORDS, RELEVANCE, twins, n_per_class=4, seed=42, scorer=fake_scorer)


def test_build_arabic_filters_and_balances():
    recs = [dict(r, id=r["id"].replace("multifc", "xfact_ar")) for r in RECORDS]
    # every snippet once with each label: the filter keeps snippet 0 as entail and
    # snippet 2 as contradict; the Y2 neutral rows are kept as they are
    ar_pairs = [dict(p, id=f"{p['id']}_{lab}", label=lab, origin="xfact_weak")
                for p in own_snippet_pairs(recs, 42, per_claim=3, language="ar")
                for lab in ("entail", "contradict", "neutral")]
    out = build_arabic(ar_pairs, recs, fake_scorer, seed=42)
    c = Counter(p["label"] for p in out)
    assert c == {"entail": 10, "contradict": 10, "neutral": 10}
    assert all(p["snippet_idx"] == 0 for p in out if p["label"] == "entail")
    assert all(p["snippet_idx"] == 2 for p in out if p["label"] == "contradict")
    assert all(p["language"] == "ar" for p in out)


def test_filter_needs_min_prob():
    rec = dict(RECORDS[0], evidence=[{"text": "snippet 5 of claim 0", "url": "", "domain": ""}])
    assert filter_by_model(own_snippet_pairs([rec], 42), fake_scorer) == []   # top 0.45


def test_run_key_differs_between_runs():
    import argparse

    from src.nli.train import run_key
    base = dict(model="joeddav/xlm-roberta-large-xnli", version="v1", train="data/nli/train.jsonl",
                seed=42, epochs=2, batch_size=16, lr=1e-5)
    v1 = run_key(argparse.Namespace(**base), 32658)
    assert v1 == run_key(argparse.Namespace(**base), 32658)
    assert v1 != run_key(argparse.Namespace(**dict(base, version="v2")), 18525)
    assert v1 != run_key(argparse.Namespace(**base), 18525)    # same names, new data
