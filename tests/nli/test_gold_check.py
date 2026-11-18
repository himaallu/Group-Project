"""Tests for src/nli/gold_check.py."""

import pandas as pd

from src.nli import gold_check as G


def test_evaluate_and_load(tmp_path):
    pd.DataFrame({"id": ["c1", "c1", "c2"], "snippet_idx": [0, 1, 0],
                  "p_entail": [0.8, 0.1, 0.1], "p_contradict": [0.1, 0.1, 0.8],
                  "p_neutral": [0.1, 0.8, 0.1]}).to_parquet(tmp_path / "x.parquet")
    gold = [{"language": "en", "claim_id": "c1", "snippet_idx": 0, "label": "entail"},
            {"language": "en", "claim_id": "c1", "snippet_idx": 1, "label": "neutral"},
            {"language": "en", "claim_id": "c2", "snippet_idx": 0, "label": "neutral"},
            {"language": "ar", "claim_id": "a1", "snippet_idx": 0, "label": "neutral"}]
    res = G.evaluate(G.load_scores(tmp_path), gold)
    assert res["en"]["n"] == 3 and res["en"]["accuracy"] == 2 / 3
    assert res["en"]["pred_counts"] == {"entail": 1, "neutral": 1, "contradict": 1}
    assert res["ar"] == {"n": 0, "n_missing": 1}
