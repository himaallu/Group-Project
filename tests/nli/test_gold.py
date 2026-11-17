"""Tests for the H2 gold-pair tools on Rathi's 10-row MultiFC val sample."""

import csv
import json
from pathlib import Path

import pytest

from src.eval.loaders import load_jsonl
from src.nli.sample_gold import build_sheet, word_overlap, write_sheet
from src.nli.sheet_to_jsonl import merge_into_gold, normalise_label, read_sheet, write_gold

SAMPLE = Path(__file__).parents[2] / "results" / "R1" / "samples" / "multifc_val_10rows.jsonl"


@pytest.fixture
def records():
    return load_jsonl(SAMPLE)


def test_word_overlap():
    assert word_overlap("Dubai Metro closed", "the dubai metro is open") == pytest.approx(2 / 3)
    assert word_overlap("", "anything") == 0.0


def test_build_sheet_is_reproducible_and_one_pair_per_claim(records):
    rows = build_sheet(records, n=6, seed=42)
    assert rows == build_sheet(records, n=6, seed=42)
    assert len(rows) == 6
    assert len({r["claim_id"] for r in rows}) == 6
    for r in rows:
        rec = next(x for x in records if x["id"] == r["claim_id"])
        assert r["snippet"] == rec["evidence"][r["snippet_idx"]]["text"]
        assert r["id"] == f"{r['claim_id']}_s{r['snippet_idx']}"
        assert r["label"] == ""


def test_build_sheet_balances_labels(records):
    # The sample has 3 supported, 5 contradicted, 2 unverified: 6 -> 2 per label.
    rows = build_sheet(records, n=6, seed=42)
    label_of = {r["id"]: r["label"] for r in records}
    counts = {}
    for r in rows:
        counts[label_of[r["claim_id"]]] = counts.get(label_of[r["claim_id"]], 0) + 1
    assert counts == {"supported": 2, "contradicted": 2, "unverified": 2}


def test_build_sheet_fills_when_a_label_is_short(records):
    rows = build_sheet(records, n=9, seed=42)   # only 2 unverified exist
    assert len(rows) == 9


def test_normalise_label():
    assert normalise_label(" E ") == "entail"
    assert normalise_label("contradiction") == "contradict"
    assert normalise_label("") == ""
    assert normalise_label("maybe") is None


def _label_sheet(tmp_path, records, labels):
    rows = build_sheet(records, n=len(labels), seed=42)
    for row, lab in zip(rows, labels):
        row["label"] = lab
    path = tmp_path / "sheet.csv"
    write_sheet(rows, path)
    return path


def test_round_trip_sheet_to_gold(tmp_path, records):
    sheet = _label_sheet(tmp_path, records, ["e", "c", "n", "neutral", "entail", "C"])
    gold = read_sheet(sheet)
    assert [g["label"] for g in gold] == ["entail", "contradict", "neutral",
                                          "neutral", "entail", "contradict"]
    out = tmp_path / "gold_val.jsonl"
    write_gold(gold, out)
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert set(first) == {"id", "claim_id", "snippet_idx", "language", "claim", "snippet", "label"}


def test_bad_and_missing_labels_are_reported(tmp_path, records):
    sheet = _label_sheet(tmp_path, records, ["e", "maybe", ""])
    with pytest.raises(ValueError, match="2 problem"):
        read_sheet(sheet)
    with pytest.raises(ValueError, match="unknown label"):
        read_sheet(sheet, allow_partial=True)


def test_partial_sheet_keeps_labelled_rows(tmp_path, records):
    sheet = _label_sheet(tmp_path, records, ["e", "", "n"])
    assert len(read_sheet(sheet, allow_partial=True)) == 2


def test_merge_keeps_arabic_rows(tmp_path, records):
    out = tmp_path / "gold_val.jsonl"
    arabic = {"id": "xfact_ar_000001_s0", "claim_id": "xfact_ar_000001", "snippet_idx": 0,
              "language": "ar", "claim": "ادعاء", "snippet": "مقتطف", "label": "neutral"}
    old_en = {**arabic, "id": "multifc_old_s0", "language": "en"}
    write_gold([arabic, old_en], out)
    new = read_sheet(_label_sheet(tmp_path, records, ["e", "c"]))
    merged = merge_into_gold(new, out)
    assert arabic in merged
    assert all(r["id"] != "multifc_old_s0" for r in merged)   # old English replaced
    assert len(merged) == 3
