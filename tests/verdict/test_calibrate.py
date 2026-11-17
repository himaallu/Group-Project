"""Tests for src/verdict/calibrate.py on synthetic logits."""

import json

import numpy as np
import pytest

from src.eval.metrics import LABELS, ece, macro_f1
from src.verdict import calibrate as C


def _overconfident(n=600, seed=42, scale=4.0):
    """Logits from a model that is right ~70% of the time but very confident."""
    rng = np.random.default_rng(seed)
    gold_idx = rng.integers(0, 3, n)
    pred_idx = np.where(rng.random(n) < 0.7, gold_idx, rng.integers(0, 3, n))
    logits = rng.normal(0, 0.3, (n, 3))
    logits[np.arange(n), pred_idx] += scale  # large margin -> over-confident
    return logits, [LABELS[i] for i in gold_idx]


def test_softmax_rows_sum_to_one_and_temperature_softens():
    logits = np.array([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    p1, p3 = C.softmax(logits, 1.0), C.softmax(logits, 3.0)
    assert np.allclose(p1.sum(axis=1), 1.0)
    assert p3[0].max() < p1[0].max()
    assert np.allclose(p1[1], 1 / 3)


def test_fit_temperature_finds_t_above_one_and_lowers_nll_and_ece():
    logits, gold = _overconfident()
    T = C.fit_temperature(logits, gold)
    assert T > 1.5
    assert C.nll(logits, gold, T) < C.nll(logits, gold, 1.0)
    assert ece(gold, C.softmax(logits, T)) < ece(gold, C.softmax(logits, 1.0))


def test_fit_temperature_recovers_known_temperature():
    # Labels sampled from softmax(z); logits given as z * 2.5 -> best T should be ~2.5.
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, (4000, 3))
    p = C.softmax(z)
    gold = [LABELS[rng.choice(3, p=row)] for row in p]
    assert C.fit_temperature(z * 2.5, gold) == pytest.approx(2.5, rel=0.1)


def test_apply_threshold():
    probs = np.array([[0.5, 0.3, 0.2], [0.9, 0.05, 0.05], [0.2, 0.7, 0.1]])
    assert C.apply_threshold(probs, 0.0) == ["supported", "supported", "contradicted"]
    assert C.apply_threshold(probs, 0.6) == ["unverified", "supported", "contradicted"]


def test_threshold_helps_when_unverified_cases_are_low_confidence():
    # Unverified claims get flat, low-confidence probabilities that never pick "unverified".
    probs = np.array([[0.9, 0.05, 0.05]] * 4 + [[0.05, 0.9, 0.05]] * 4
                     + [[0.4, 0.35, 0.25]] * 4)
    gold = ["supported"] * 4 + ["contradicted"] * 4 + ["unverified"] * 4
    tau = C.fit_unverified_threshold(probs, gold)
    assert 0.4 < tau <= 0.9
    assert macro_f1(gold, C.apply_threshold(probs, tau)) == pytest.approx(1.0)


def test_threshold_zero_when_it_does_not_help():
    probs = np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]])
    assert C.fit_unverified_threshold(probs, list(LABELS)) == 0.0


def test_fit_save_load_roundtrip_and_cli(tmp_path, monkeypatch):
    logits, gold = _overconfident(n=60)
    path = tmp_path / "val_logits.jsonl"
    with open(path, "w") as f:
        for i, (row, g) in enumerate(zip(logits, gold)):
            f.write(json.dumps({"id": f"multifc_{i:06d}", "gold": g, "logits": row.tolist()}) + "\n")
    out = tmp_path / "calib.json"
    monkeypatch.setattr("sys.argv", ["calibrate", "--val-logits", str(path), "--out", str(out)])
    C.main()
    s = C.load_settings(out)
    assert s["T"] > 1 and s["n_val"] == 60
    assert s["val_nll_after"] <= s["val_nll_before"]
    preds, probs = C.apply_calibration(logits, s["T"], s["tau"])
    assert len(preds) == 60 and np.allclose(probs.sum(axis=1), 1.0)
