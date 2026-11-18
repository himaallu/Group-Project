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


def test_main_table_mean_sd_and_calibrated_ece():
    from src.eval.figures import main_table, pct
    summary = pd.DataFrame([
        {"model": "twin", "macro_f1_mean": 0.3518, "macro_f1_std": 0.0069, "esr_mean": 0.8968,
         "esr_std": 0.0094, "ece_mean": 0.0816, "ece_std": 0.0105},
        {"model": "twin_cal", "macro_f1_mean": 0.3518, "macro_f1_std": 0.0069, "ece_mean": 0.0699,
         "ece_std": 0.0032},
        {"model": "avt_twin", "macro_f1_mean": 0.3828, "macro_f1_std": 0.0199},
    ])
    df = main_table(summary)
    assert list(df["model"]) == ["twin loss"]
    assert df.loc[0, "ESR"] == "89.7 ± 0.9" and df.loc[0, "ECE (cal.)"] == "7.0 ± 0.3"
    avt = main_table(summary, prefix="avt_")
    assert "ESR" not in avt.columns                     # no twins on AVeriTeC
    assert pct(0.3625) == "36.3"                        # rounded half up


def test_nli_table_one_zero_shot_row_per_base_model():
    from src.eval.figures import nli_table
    ev = {"zero_shot": {"gold_en": {"macro_f1": 0.37}, "val_weak": {"macro_f1": 0.9}},
          "fine_tuned": {"gold_en": {"macro_f1": 0.33}}}
    df = nli_table([("m", "v1", ev), ("m", "v2", ev)])
    assert list(df["setting"]) == ["zero-shot", "fine-tuned v1", "fine-tuned v2"]
    assert "val_weak macro-F1" not in df.columns


def test_bootstrap_table_in_points():
    from src.eval.figures import bootstrap_table
    boot = {"preds_twin_s42 vs preds_notwin_s42": {
        "esr": {"diff": 0.352, "ci_low": 0.292, "ci_high": 0.413, "p_value": 0.0}}}
    row = bootstrap_table(boot).iloc[0]
    assert row["seed"] == "42" and row["diff"] == 35.2 and row["95% CI"] == "[29.2, 41.3]"
