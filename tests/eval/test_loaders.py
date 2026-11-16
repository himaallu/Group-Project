"""Tests for src/eval/loaders.py: format checks, twin ids, joining."""

import copy
import json
from pathlib import Path

import pytest

from src.eval.loaders import (join_gold, load_preds, split_twin_id, summarise_file,
                              twin_type, validate_pred, validate_record)

GOOD = {"id": "multifc_000123", "source": "multifc", "split": "train", "language": "en",
        "claim": "Dubai Metro suspended all lines today because of flooding",
        "evidence": [{"text": "Green Line operating normally, RTA confirms.",
                      "url": "https://example.com", "domain": "gulfnews.com"}],
        "image_path": None, "label": "contradicted", "orig_label": "false"}


def test_valid_record_has_no_problems():
    assert validate_record(GOOD) == []


@pytest.mark.parametrize("change, expected_word", [
    ({"label": "false"}, "label"),
    ({"language": "arabic"}, "language"),
    ({"split": "dev"}, "split"),
    ({"id": "averitec_1"}, "does not start with source"),
    ({"evidence": [{"text": "x", "url": "u"}]}, "domain"),
    ({"evidence": "not a list"}, "evidence must be a list"),
    ({"claim": "  "}, "claim"),
])
def test_broken_record_is_reported(change, expected_word):
    rec = {**copy.deepcopy(GOOD), **change}
    problems = validate_record(rec)
    assert any(expected_word in p for p in problems), problems


def test_missing_key_is_reported():
    rec = copy.deepcopy(GOOD)
    del rec["orig_label"]
    assert any("missing keys" in p for p in validate_record(rec))


@pytest.mark.parametrize("rid, base, suffixes, kind", [
    ("multifc_000123", "multifc_000123", (), None),
    ("multifc_000123_P", "multifc_000123", ("P",), "P"),
    ("multifc_000123_B_A", "multifc_000123", ("B", "A"), "B"),   # Arabic swap counts as B
    ("xfact_ar_000007_C", "xfact_ar_000007", ("C",), "C"),
    ("verite_000004_V", "verite_000004", ("V",), "V"),
])
def test_split_twin_id(rid, base, suffixes, kind):
    assert split_twin_id(rid) == (base, suffixes)
    assert twin_type(rid) == kind


def test_validate_pred():
    good = {"id": "x", "pred_label": "supported",
            "probs": {"supported": 0.7, "contradicted": 0.2, "unverified": 0.1}}
    assert validate_pred(good) == []
    bad = {**good, "probs": {"supported": 0.7, "contradicted": 0.7, "unverified": 0.1}}
    assert any("sum" in p for p in validate_pred(bad))
    assert validate_pred({**good, "pred_label": "true"})


def test_load_preds_rejects_duplicates(tmp_path):
    p = {"id": "x", "pred_label": "supported",
         "probs": {"supported": 1.0, "contradicted": 0.0, "unverified": 0.0}}
    path = tmp_path / "preds_dup.jsonl"
    path.write_text(json.dumps(p) + "\n" + json.dumps(p) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_preds(path)


def test_join_gold_reports_missing(fake_preds, fake_gold):
    extra = {"id": "multifc_999999", "pred_label": "supported",
             "probs": {"supported": 1.0, "contradicted": 0.0, "unverified": 0.0}}
    df, missing = join_gold(fake_preds + [extra], fake_gold)
    assert missing == ["multifc_999999"]
    assert len(df) == 27
    row = df.set_index("id").loc["multifc_000003_B_A"]
    assert (row["base_id"], row["twin_type"], row["language"]) == ("multifc_000003", "B", "ar")


def test_summarise_fake_file():
    s = summarise_file(Path(__file__).parents[1] / "fixtures" / "fake_gold.jsonl")
    assert s["n_records"] == 27 and s["n_invalid"] == 0
    assert s["twin_type"]["B"] == 5
