"""Tests for src/pipeline.py: the check() interface shape with stand-in steps."""

import json

import numpy as np
import pytest

from src import pipeline as P
from src.eval.metrics import LABELS

FIELDS = {"claim", "language", "label", "confidence", "probs", "evidence", "explanation"}


@pytest.fixture(autouse=True)
def stand_ins_only(tmp_path, monkeypatch):
    """Use the stand-in steps: no calibration file and no teammate/real modules
    (those would download large models); tests that need one patch it in."""
    monkeypatch.setattr(P, "CALIB_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(P, "_optional", lambda module, attr: None)


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


# ---------- the real verdict head (src/verdict/infer.py) with fake heavy models ----------

class _FakeReranker:
    """Stands in for Rathi's reranker: relevance 0.9 for the first snippet, 0.05 after."""

    def score(self, claim, snippets):
        return [0.9] + [0.05] * (len(snippets) - 1)


class _FakeEnt:
    def __init__(self, text, label):
        self.text, self.label_, self.start_char, self.end_char = text, label, 0, len(text)


def _fake_nlp(text):
    """Stands in for spaCy: finds 'Dubai' as a place."""
    class Doc:
        ents = [_FakeEnt("Dubai", "GPE")] if "Dubai" in text else []
    return Doc()


@pytest.fixture
def infer(monkeypatch):
    pytest.importorskip("torch")
    from src.verdict import infer as I
    I._head.cache_clear()
    monkeypatch.setattr(I, "_reranker", lambda: _FakeReranker())
    monkeypatch.setattr(I, "_spacy", lambda: _fake_nlp)
    return I


def test_predict_one_with_trained_head(infer):
    nli = [{"p_entail": 0.05, "p_contradict": 0.9, "p_neutral": 0.05},
           {"p_entail": 0.1, "p_contradict": 0.1, "p_neutral": 0.8}]
    snippets = [{"text": "Dubai Metro Green Line operating normally."}, {"text": "Unrelated."}]
    probs = infer.predict_one("Dubai Metro suspended all lines today", snippets, nli)
    assert probs.shape == (3,) and probs.sum() == pytest.approx(1.0)


def test_relevance_from_snippets_skips_reranker(infer, monkeypatch):
    monkeypatch.setattr(infer, "_reranker", lambda: pytest.fail("reranker should not be called"))
    rows = infer.relevance_rows("c", [{"text": "a", "relevance": 0.3}, {"text": "b", "relevance": 0.7}])
    assert rows == {0: (0.3,), 1: (0.7,)}


def test_arabic_claims_skip_english_entities(infer):
    assert infer.claim_entities("مترو دبي متوقف") == []
    assert infer.claim_entities("Dubai Metro is closed")[0]["label"] == "GPE"


def test_pipeline_uses_trained_head(infer, monkeypatch):
    real = {("src.verdict.infer", "predict_one"): infer.predict_one}
    monkeypatch.setattr(P, "_optional", lambda m, a: real.get((m, a)))
    assert P._verdict_head() is not None
    out = P.check("Dubai Metro suspended all lines today.",
                  evidence=["Dubai Metro Green Line operating normally, RTA confirms."])
    _check_result(out[0])
    assert P.component_status()["verdict (H3)"] == "real"


def test_verdict_head_needs_model_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "_optional", lambda m, a: (lambda *x: None))
    monkeypatch.setenv("TWINCHECK_VERDICT", str(tmp_path / "missing"))
    assert P._verdict_head() is None
