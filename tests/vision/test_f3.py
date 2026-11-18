"""Tests for the F3 probe and leakage helpers (tiny fake data, no downloads).

Run: python -m pytest tests/vision -q
"""
import numpy as np
import pandas as pd

from src.vision.leakage import features, to_preds
from src.vision.probe import evaluate, train_probe, twin_summary


def test_probe_separates_fake_from_real():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (40, 8)), rng.normal(3, 1, (40, 8))])
    meta = pd.DataFrame({"class_name": ["real"] * 40 + ["fake"] * 40,
                         "generator": ["real"] * 40 + ["SD"] * 40})
    probe = train_probe(X, (meta["class_name"] == "fake").astype(int).to_numpy())
    result = evaluate(probe, X, meta)
    assert result["auroc"] > 0.95 and set(result) >= {"acc_real", "acc_SD"}


def test_twin_summary_pairs_originals_with_twins():
    scores = pd.DataFrame({"dataset": ["VERITE_VAE"] * 4, "image_id": ["a", "a", "b", "b"],
                           "role": ["original", "twin"] * 2, "p_ai": [0.1, 0.6, 0.2, 0.1]})
    row = twin_summary(scores).iloc[0]
    assert row["n_pairs"] == 2 and row["twin_higher_share"] == 0.5
    assert row["flagged_ai_twin"] == 0.5


def test_decoupled_model_never_sees_authenticity():
    img, cap = np.ones((3, 4)), np.full((3, 4), 0.5)
    low, high = features("decoupled", img, cap, np.zeros(3)), features("decoupled", img, cap, np.ones(3))
    assert np.array_equal(low, high)                       # p_ai has no effect
    assert features("fused", img, cap, np.ones(3)).shape == (3, 13)
    assert features("fused_no_auth", img, cap, np.ones(3)).shape == (3, 12)


def test_preds_follow_the_shared_format():
    class Fixed:
        def predict_proba(self, X):
            return np.array([[0.7, 0.3]] * len(X))
    preds = to_preds(["verite_000001"], Fixed(), np.zeros((1, 2)))
    assert preds[0]["pred_label"] == "supported"
    assert abs(sum(preds[0]["probs"].values()) - 1) < 1e-9
