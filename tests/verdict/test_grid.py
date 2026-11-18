"""Tests for src/verdict/grid.py (λ selection, end-to-end grid) and calibrate_pred_file."""

import json

import pytest

pytest.importorskip("torch")

from src.verdict import calibrate as C  # noqa: E402
from src.verdict import grid as G  # noqa: E402
from tests.verdict.test_h3 import _synthetic  # noqa: E402


def _row(li, ls, f1, score):
    return {"name": G.run_name(li, ls), "lambda_inv": li, "lambda_sens": ls,
            "val_macro_f1": f1, "val_twin_score": score}


def test_run_name():
    assert G.run_name(0.5, 1.0) == "inv0.5_sens1" and G.run_name(0, 0) == "inv0_sens0"


def test_select_respects_f1_floor_and_prefers_twin_score():
    rows = [_row(0, 0, 0.40, 0.50), _row(1, 1, 0.395, 0.80),
            _row(2, 1, 0.35, 0.95)]          # best twin score but loses 5 F1 points
    choice = G.select(rows)
    assert choice["chosen"]["name"] == "inv1_sens1"
    assert "inv2_sens1" not in choice["eligible"]


def test_select_ties_go_to_smaller_lambda_and_base_required():
    rows = [_row(0, 0, 0.40, 0.5), _row(1, 1, 0.40, 0.9), _row(0.5, 0.5, 0.40, 0.9)]
    assert G.select(rows)["chosen"]["name"] == "inv0.5_sens0.5"
    with pytest.raises(ValueError):
        G.select([_row(1, 1, 0.4, 0.9)])


def test_twin_score_skips_missing():
    assert G.twin_score({"esr": 0.5, "unrelated_rate": 1.0, "paraphrase_consistency": None}) == 0.75
    assert G.twin_score({}) is None


def test_grid_end_to_end(tmp_path):
    p = _synthetic(tmp_path)
    argv = ["--out-dir", str(tmp_path / "H4"), "--models-dir", str(tmp_path / "models"),
            "--lambda-inv", "0", "1", "--lambda-sens", "0", "1", "--seeds", "42",
            "--twins", str(p["twins"]), "--train", str(p["train"]), "--val", str(p["val"]),
            "--test", str(p["test"]), "--nli", str(p["nli"]), "--relevance", str(p["rel"]),
            "--entities", str(p["ent"]), "--epochs", "5", "--batch-groups", "8", "--no-wandb"]
    choice = G.main(argv)
    out = tmp_path / "H4"
    assert (out / "grid_summary.csv").read_text().count("\n") == 5   # header + 4 runs
    assert json.loads((out / "selected_lambda.json").read_text())["chosen"]["name"] == choice["chosen"]["name"]
    for name in ("twin", "notwin"):
        assert (out / f"preds_{name}_s42.jsonl").exists()
        assert (tmp_path / "models" / f"verdict_{name}_s42" / "model.pt").exists()
    assert not list(out.glob("preds_inv*"))        # grid runs stay in grid/


def test_grid_requires_twins():
    with pytest.raises(SystemExit):
        G.main(["--nli", "x.parquet"])


def test_calibrate_pred_file(tmp_path):
    src = tmp_path / "p.jsonl"
    src.write_text(json.dumps({"id": "a", "pred_label": "supported",
                               "probs": {"supported": 0.5, "contradicted": 0.3, "unverified": 0.2}}) + "\n")
    n = C.calibrate_pred_file({"T": 1.0, "tau": 0.6}, src, tmp_path / "out.jsonl")
    row = json.loads((tmp_path / "out.jsonl").read_text())
    assert n == 1 and row["pred_label"] == "unverified"          # 0.5 < tau
    assert row["probs"]["supported"] == pytest.approx(0.5, abs=1e-5)  # T = 1 keeps probs
    C.calibrate_pred_file({"T": 3.0, "tau": 0.0}, src, tmp_path / "soft.jsonl")
    soft = json.loads((tmp_path / "soft.jsonl").read_text())
    assert soft["pred_label"] == "supported" and soft["probs"]["supported"] < 0.5
