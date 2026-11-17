"""Tests for src/pipeline.py: the check() interface shape with stand-in steps."""

import json

import numpy as np
import pytest

from src import pipeline as P
from src.eval.metrics import LABELS

FIELDS = {"claim", "language", "label", "confidence", "probs", "evidence", "explanation"}


@pytest.fixture(autouse=True)
def no_calibration_file(tmp_path, monkeypatch):
    """Tests must not depend on a real models/verdict_calib.json."""
    monkeypatch.setattr(P, "CALIB_PATH", tmp_path / "missing.json")


def _check_result(r):
    assert set(r) == FIELDS
    assert r["label"] in LABELS and r["language"] in ("en", "ar", "mixed")
    assert sum(r["probs"].values()) == pytest.approx(1.0)
    assert r["confidence"] == pytest.approx(r["probs"][r["label"]])


def test_detect_language():
    assert P.detect_language("Dubai Metro is closed today") == "en"
    assert P.detect_language("مترو دبي مغلق اليوم") == "ar"
    assert P.detect_language("مترو دبي مغلق today بسبب flooding in the city") == "mixed"
    assert P.detect_language("123 !!") == "en"


def test_check_english_two_claims_with_evidence():
    msg = "Dubai Metro suspended all lines today. The RTA said the Green Line runs normally."
    ev = ["Green Line operating normally, RTA confirms.",
          {"text": "Dubai Metro suspended lines today after flooding.", "url": "https://x", "domain": "x.com"}]
    out = P.check(msg, evidence=ev)
    assert len(out) == 2
    for r in out:
        _check_result(r)
        assert len(r["evidence"]) == 2
        assert {"text", "url", "domain", "p_entail", "p_contradict", "p_neutral"} <= set(r["evidence"][0])
        assert "[" in r["explanation"]


def test_check_without_evidence_is_unverified():
    out = P.check("مترو دبي أوقف جميع الخطوط اليوم بسبب الفيضانات")
    assert len(out) == 1
    _check_result(out[0])
    assert out[0]["language"] == "ar"
    assert out[0]["label"] == "unverified" and out[0]["evidence"] == []


def test_lang_override_and_empty_message():
    out = P.check("مترو دبي suspended all lines today", lang="mixed")
    assert out[0]["language"] == "mixed"
    assert P.check("   ") == []


def test_calibration_file_is_applied(tmp_path, monkeypatch):
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"T": 1.0, "tau": 0.99}))
    monkeypatch.setattr(P, "CALIB_PATH", calib)
    label, probs = P.calibrate(np.array([0.6, 0.3, 0.1]))
    assert label == "unverified"           # max prob 0.6 < tau 0.99
    assert probs == pytest.approx([0.6, 0.3, 0.1])


def test_real_component_is_used_when_present(monkeypatch):
    monkeypatch.setattr(P, "_optional", lambda m, a: (lambda msg: ["Claim one here.", "Claim two here."])
                        if m == "src.claims.extract" else None)
    assert [r["claim"] for r in P.check("anything at all goes")] == ["Claim one here.", "Claim two here."]


def test_component_status_lists_all_steps():
    status = P.component_status()
    assert len(status) == 6 and set(status.values()) <= {"real", "stub"}
