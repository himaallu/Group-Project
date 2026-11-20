"""Tests for the E7 scoring in src/eval/run_y5_ar.py (no models needed)."""

import pandas as pd

from src.eval import run_y5_ar as y5


def row(i, gold, pred):
    base = i.split("_")[0] + "_" + i.split("_")[1]
    return {"id": i, "base_id": base, "gold": gold, "pred": pred}


def test_kind_and_group():
    assert y5.kind("multifc_1") == "orig"
    assert y5.kind("multifc_1_B_A") == "B_A"
    assert y5.group_of("multifc_1_M", "twins") == "mixed"
    assert y5.group_of("multifc_1_C", "twins") == "en"
    assert y5.group_of("xfact_ar_1", "xfact_ar_test") == "ar"


def test_scores_per_language():
    df = pd.DataFrame([
        row("multifc_1", "supported", "supported"),
        row("multifc_1_B", "contradicted", "contradicted"),
        row("multifc_1_C", "unverified", "unverified"),
        row("multifc_1_A", "supported", "contradicted"),
        row("multifc_1_B_A", "contradicted", "contradicted"),
        row("multifc_1_C_A", "unverified", "supported"),
        row("multifc_1_M", "supported", "supported"),
    ])
    s = {r["language"]: r for r in y5.scores(df, "twins")}
    assert s["en"]["accuracy"] == 1.0 and s["en"]["esr"] == 1.0 and s["en"]["crossling"] is None
    assert s["ar"]["esr"] == 1.0 and s["ar"]["unrelated"] == 0.0 and s["ar"]["crossling"] == 0.0
    assert s["mixed"]["crossling"] == 1.0 and s["mixed"]["macro_f1"] is None
