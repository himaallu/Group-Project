"""Tests for the F3 probe helpers (tiny fake data, no downloads).

Run: python -m pytest tests/vision -q
"""
import numpy as np
import pandas as pd

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
