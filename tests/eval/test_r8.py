"""Unit tests for R8 (T1 sheet sampling, report tables and figures) on toy rows.

Run: python -m pytest tests/eval/test_r8.py -q
"""

import json

import pandas as pd

from src.eval.figures import build_all, fmt, to_latex, twins_table
from src.eval.t1_sheet import sample_notes, sheet_rows


def toy_notes():
    """6 English and 4 Arabic notes."""
    return [{"id": f"n{i}", "language": "en" if i < 6 else "ar", "claim": f"c{i}",
             "label": "supported", "note": "x [1].",
             "snippets": [{"text": "s1"}, {"text": "s2"}]} for i in range(10)]


def test_t1_sample_and_raters():
    notes = sample_notes(toy_notes(), n_en=4, n_ar=2)
    assert [n["language"] for n in notes].count("en") == 4
    assert notes == sample_notes(toy_notes(), n_en=4, n_ar=2)      # reproducible
    rows = sheet_rows(notes)
    assert len(rows) == 4 * 2 + 2                                  # EN rated twice, AR once
    assert {r["rater"] for r in rows if r["language"] == "ar"} == {"youssef"}
    assert rows[0]["snippet_3"] == "" and rows[0]["verdict_correct"] == ""


def test_latex_and_formatting():
    df = pd.DataFrame([{"model": "a_b", "F1 %": 0.12345, "n": 3}])
    tex = to_latex(df, "Caption & more", "tab:x")
    assert r"a\_b" in tex and r"F1 \%" in tex and "0.123" in tex and r"\toprule" in tex
    assert fmt(float("nan")) == "–" and fmt(7) == "7"


def test_build_all_from_toy_results(tmp_path):
    res = tmp_path / "results"
    (res / "R3").mkdir(parents=True)
    report = {s: {"n_sources": 5, "P_made": 2, "B_made": 3, "C_made": 5, "n_twins": 10}
              for s in ("test", "train")}
    (res / "R3" / "make_twins_report.json").write_text(json.dumps(report))
    assert list(twins_table(report)["_B"]) == [3, 3]
    build_all(res, tmp_path / "tables", tmp_path / "figures", [])
    assert (tmp_path / "tables" / "tab_twins.tex").exists()
    assert (tmp_path / "figures" / "fig_twin_counts.png").stat().st_size > 1000
