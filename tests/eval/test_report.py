"""Tests for src/eval/report.py using the fake fixtures in a temporary results/ folder."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.eval.report import build_report, parse_pred_path

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_parse_pred_path():
    assert parse_pred_path(Path("results/H5/preds_ours_s43.jsonl")) == ("H5", "ours", 43)
    assert parse_pred_path(Path("results/F4/preds_claim_only.jsonl")) == ("F4", "claim_only", None)


@pytest.fixture
def results_dir(tmp_path):
    """results/H5 with the fake preds as two seeds, plus results/F4 with one baseline."""
    h5 = tmp_path / "results" / "H5"
    h5.mkdir(parents=True)
    shutil.copy(FIXTURES / "fake_preds.jsonl", h5 / "preds_ours_s42.jsonl")
    # Seed 43: same, but every _B twin predicted contradicted (ESR 1.0).
    with open(FIXTURES / "fake_preds.jsonl") as f_in, open(h5 / "preds_ours_s43.jsonl", "w") as f_out:
        for line in f_in:
            p = json.loads(line)
            if "_B" in p["id"]:
                p["pred_label"] = "contradicted"
                p["probs"] = {"supported": 0.1, "contradicted": 0.8, "unverified": 0.1}
            f_out.write(json.dumps(p) + "\n")
    f4 = tmp_path / "results" / "F4"
    f4.mkdir()
    shutil.copy(FIXTURES / "fake_preds.jsonl", f4 / "preds_claim_only.jsonl")
    return tmp_path / "results"


def test_build_report_writes_all_tables(results_dir, tmp_path):
    out = tmp_path / "tables"
    per_file = build_report(results_dir, [FIXTURES / "fake_gold.jsonl"], out)
    assert len(per_file) == 3
    for name in ("metrics_all.csv", "metrics_by_model.csv", "main_results.tex", "twin_results.tex"):
        assert (out / name).exists()

    by_model = pd.read_csv(out / "metrics_by_model.csv").set_index("model")
    ours = by_model.loc["ours"]
    assert ours["n_seeds"] == 2
    assert ours["esr_mean"] == pytest.approx(0.9)          # mean of 0.8 and 1.0
    assert by_model.loc["claim_only", "n_seeds"] == 1

    twin_tex = (out / "twin_results.tex").read_text()
    assert "claim\\_only" in twin_tex                       # underscore escaped
    assert "90.0 $\\pm$ 14.1" in twin_tex                   # ESR mean ± sd in %


def test_missing_gold_file_raises(results_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        build_report(results_dir, [tmp_path / "nope.jsonl"], tmp_path / "out")


def test_smoke_test_runs_on_fixture(tmp_path):
    from src.eval.smoke_test import run
    metrics = run([FIXTURES / "fake_gold.jsonl"], tmp_path)
    assert metrics["n_orig"] == 10 and metrics["n_B"] == 5
    assert (tmp_path / "preds_random.jsonl").exists()
