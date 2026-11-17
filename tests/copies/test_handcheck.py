"""Tests for src/copies/handcheck_sheet.py and handcheck_kappa.py on fake twins
made from Rathi's 10-row MultiFC sample."""

import copy
import csv
import json

import pytest

from src.copies import handcheck_kappa as K
from src.copies import handcheck_sheet as S
from src.eval.loaders import load_jsonl

SAMPLE = "results/R1/samples/multifc_val_10rows.jsonl"


def _fake_data(copies_per_type=1):
    """10 originals (relabelled as test) + _P, _B, _C twins for each."""
    originals = load_jsonl(SAMPLE)[:10]
    for o in originals:
        o["split"] = "test"
    twins = []
    for k in range(copies_per_type):
        for i, o in enumerate(originals):
            tag = "" if k == 0 else str(k)  # extra copies get unique ids
            p = copy.deepcopy(o); p["id"] = f"{o['id']}{tag}_P"; p["claim"] = "Reworded: " + o["claim"]
            b = copy.deepcopy(o); b["id"] = f"{o['id']}{tag}_B"; b["label"] = "contradicted"
            b["evidence"][0]["text"] = "SWAPPED " + b["evidence"][0]["text"]
            c = copy.deepcopy(o); c["id"] = f"{o['id']}{tag}_C"; c["label"] = "unverified"
            c["evidence"] = copy.deepcopy(originals[(i + 1) % len(originals)]["evidence"])
            twins += [p, b, c]
    # extra copies need originals with matching base ids
    extra = []
    for k in range(1, copies_per_type):
        for o in originals:
            e = copy.deepcopy(o); e["id"] = f"{o['id']}{k}"; extra.append(e)
    return originals + extra, twins


def test_expected_label_rules():
    assert S.expected_label("B", "supported") == "contradicted"
    assert S.expected_label("C", "supported") == "unverified"
    assert S.expected_label("P", "supported") == "supported"


def test_evidence_diff_shows_only_changed_snippets():
    o = [{"text": "a"}, {"text": "b"}]
    assert S.evidence_diff(o, [{"text": "a"}, {"text": "B2"}]) == ("b", "B2")
    assert S.evidence_diff(o, o)[1] == "(same evidence)"


def test_build_sheets_sizes_overlap_balance_and_seed():
    originals, twins = _fake_data()           # 30 twins: 10 P, 10 B, 10 C
    sheets = S.build_sheets(twins, originals, n_per_rater=12, overlap=6, seed=42)
    h, f = sheets["himasri"], sheets["feba"]
    assert len(h) == len(f) == 12
    shared = {r["id"] for r in h} & {r["id"] for r in f}
    assert len(shared) == 6
    kinds = [r["twin_type"] for r in h + f if r["id"] not in shared]
    kinds += [r["twin_type"] for r in h if r["id"] in shared]
    assert {k: kinds.count(k) for k in "PBC"} == {"P": 6, "B": 6, "C": 6}
    assert S.build_sheets(twins, originals, 12, 6, 42) == sheets  # reproducible
    row = next(r for r in h if r["twin_type"] == "B")
    assert row["expected_label"] == "contradicted" and row["twin_evidence"].startswith("SWAPPED")


def test_default_sizes_give_120_distinct_twins():
    originals, twins = _fake_data(copies_per_type=5)   # 150 twins
    sheets = S.build_sheets(twins, originals)
    ids = {r["id"] for rows in sheets.values() for r in rows}
    assert len(sheets["himasri"]) == 75 and len(ids) == 120


def test_build_sheets_refuses_non_test_and_orphans():
    originals, twins = _fake_data()
    twins[0]["split"] = "val"
    with pytest.raises(ValueError, match="test twins only"):
        S.build_sheets(twins, originals, 12, 6)
    originals, twins = _fake_data()
    with pytest.raises(ValueError, match="no original"):
        S.build_sheets(twins, originals[1:], 12, 6)


def _fill(path, rows, answers):
    """Write a sheet with the given valid answers (dict id -> y/n)."""
    for r in rows:
        r["valid"] = answers.get(r["id"], "y")
    S.write_sheet(rows, path)


def test_kappa_end_to_end(tmp_path, monkeypatch):
    originals, twins = _fake_data()
    sheets = S.build_sheets(twins, originals, n_per_rater=12, overlap=6, seed=42)
    shared = sorted({r["id"] for r in sheets["himasri"]} & {r["id"] for r in sheets["feba"]})
    # Feba says n on two shared twins, Himasri on one of those -> 1 disagreement
    _fill(tmp_path / "h.csv", sheets["himasri"], {shared[0]: "n"})
    _fill(tmp_path / "f.csv", sheets["feba"], {shared[0]: "N", shared[1]: "no"})
    out = tmp_path / "summary.json"
    monkeypatch.setattr("sys.argv", ["k", str(tmp_path / "h.csv"), str(tmp_path / "f.csv"),
                                     "--out", str(out)])
    K.main()
    s = json.loads(out.read_text())
    assert s["n_overlap"] == 6 and s["n_rated"] == 18
    assert s["disagreements"] == [shared[1]]
    assert s["agreement"] == pytest.approx(5 / 6)
    # p_o = 5/6; Himasri n-rate 1/6, Feba 2/6 -> p_e = 1/18 + 10/18 = 11/18
    assert s["kappa"] == pytest.approx((5 / 6 - 11 / 18) / (1 - 11 / 18))
    assert s["valid_rate"]["all"] == pytest.approx(16 / 18)


def test_read_sheet_blanks_and_bad_values(tmp_path):
    path = tmp_path / "s.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "twin_type", "valid"])
        w.writeheader()
        w.writerows([{"id": "x_P", "twin_type": "P", "valid": "y"},
                     {"id": "y_B", "twin_type": "B", "valid": ""}])
    with pytest.raises(ValueError, match="not rated"):
        K.read_sheet(path)
    assert list(K.read_sheet(path, allow_partial=True)) == ["x_P"]
    with pytest.raises(ValueError):
        K.normalise_valid("maybe")
