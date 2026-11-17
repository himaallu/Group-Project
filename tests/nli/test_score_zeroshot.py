"""Tests for src/nli/score_zeroshot.py (pair building and label mapping need no model)."""

import pytest

from src.eval.loaders import load_jsonl
from src.nli import score_zeroshot as Z

SAMPLE = "results/R1/samples/multifc_val_10rows.jsonl"


def test_build_pairs_one_row_per_nonempty_snippet():
    records = load_jsonl(SAMPLE)
    records[0]["evidence"].append({"text": "  ", "url": None, "domain": None})
    pairs = Z.build_pairs(records)
    expected = sum(1 for r in records for e in r["evidence"] if e["text"].strip())
    assert len(pairs) == expected
    first = pairs.iloc[0]
    assert first["hypothesis"] == records[0]["claim"]
    assert first["premise"] == records[0]["evidence"][0]["text"].strip()
    assert list(pairs["snippet_idx"][:2]) == [0, 1]


def test_build_pairs_empty():
    assert Z.build_pairs([{"id": "x", "claim": "c", "evidence": []}]).empty


def test_label_columns_any_order():
    assert Z.label_columns({0: "entailment", 1: "neutral", 2: "contradiction"}) == \
        {"p_entail": 0, "p_neutral": 1, "p_contradict": 2}
    assert Z.label_columns({"0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL"}) == \
        {"p_contradict": 0, "p_entail": 1, "p_neutral": 2}
    with pytest.raises(ValueError):
        Z.label_columns({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
