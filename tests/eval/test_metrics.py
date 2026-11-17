"""Tests for src/eval/metrics.py on the fake fixtures (expected values in
tests/fixtures/README.md) and on small hand-made cases."""

import numpy as np
import pandas as pd
import pytest

from src.eval import metrics as M


# ---------- metrics on the fake fixture file ----------

def test_macro_f1_on_fixture(fake_df):
    orig = M.originals(fake_df)
    assert M.macro_f1(list(orig["gold"]), list(orig["pred"])) == pytest.approx(0.69444, abs=1e-4)


def test_ece_on_fixture(fake_df):
    orig = M.originals(fake_df)
    probs = orig[list(M.PROB_COLS)].to_numpy()
    assert M.ece(orig["gold"], probs) == pytest.approx(0.10)


def test_b_twin_predicted_contradicted_counts_toward_esr(fake_df):
    # 5 _B twins (one Arabic, multifc_000003_B_A), 4 predicted contradicted.
    assert M.evidence_sensitivity(fake_df) == pytest.approx(0.80)


def test_unrelated_rate_on_fixture(fake_df):
    assert M.unrelated_rate(fake_df) == pytest.approx(0.50)


def test_paraphrase_consistency_compares_with_original_prediction(fake_df):
    # multifc_000004 is predicted contradicted (wrong), its _P supported -> inconsistent.
    assert M.paraphrase_consistency(fake_df) == pytest.approx(0.75)


def test_crosslingual_consistency_on_fixture(fake_df):
    assert M.crosslingual_consistency(fake_df) == pytest.approx(0.75)
    assert M.crosslingual_consistency(fake_df, ("A",)) == pytest.approx(1.0)
    assert M.crosslingual_consistency(fake_df, ("M",)) == pytest.approx(0.5)


def test_alr_on_fixture(fake_df):
    assert M.authenticity_leakage(fake_df) == pytest.approx(0.50)


def test_compute_all_on_fixture(fake_df):
    r = M.compute_all(fake_df)
    assert r["n_orig"] == 10 and r["n_B"] == 5 and r["n_V"] == 2
    assert r["esr"] == pytest.approx(0.8)
    assert 0.0 <= r["auroc"] <= 1.0


# ---------- edge cases ----------

def test_twin_metrics_are_none_without_twins(fake_df):
    only_orig = M.originals(fake_df)
    r = M.compute_all(only_orig)
    for key in ("esr", "unrelated_rate", "paraphrase_consistency",
                "crosslingual_consistency", "alr"):
        assert r[key] is None


def test_twin_without_its_original_is_skipped(fake_df):
    df = fake_df[fake_df["id"] != "multifc_000004"]   # drop the original of the bad _P
    assert M.paraphrase_consistency(df) == pytest.approx(1.0)


def test_empty_inputs_return_none():
    assert M.macro_f1([], []) is None
    assert M.ece([], np.zeros((0, 3))) is None
    assert M.faithfulness([]) is None
    assert M.citation_accuracy([]) is None


# ---------- calibration and AUROC ----------

def test_ece_perfectly_calibrated_is_zero():
    # Confidence 1.0 and always right.
    probs = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    assert M.ece(["supported", "contradicted"], probs) == pytest.approx(0.0)


def test_ece_uses_given_prediction_not_argmax():
    # Unverified threshold turned an argmax-"supported" into "unverified" (correct).
    probs = np.array([[0.5, 0.2, 0.3]])
    assert M.ece(["unverified"], probs, pred=["unverified"]) == pytest.approx(0.7)
    assert M.ece(["unverified"], probs) == pytest.approx(0.5)


def test_auroc_perfect_and_skips_missing_class():
    gold = ["supported", "supported", "contradicted", "contradicted"]
    probs = np.array([[0.9, 0.1, 0], [0.8, 0.2, 0], [0.1, 0.9, 0], [0.3, 0.7, 0]])
    assert M.auroc(gold, probs) == pytest.approx(1.0)   # "unverified" absent -> skipped
    assert M.auroc(["supported"] * 3, probs[:3]) is None


# ---------- explanation metrics ----------

def test_faithfulness_and_citation_accuracy():
    assert M.faithfulness([[True, False], [True]]) == pytest.approx(2 / 3)
    assert M.citation_accuracy([True, True, False, True]) == pytest.approx(0.75)


# ---------- claim extraction and retrieval ----------

def _exact_sim(preds, golds):
    """Fake similarity: 1 if the strings are equal (case-insensitive), else 0.5."""
    return np.array([[1.0 if p.lower() == g.lower() else 0.5 for g in golds] for p in preds])


def test_claim_extraction_f1():
    pred = [["A", "B", "X"], [], ["C"]]
    gold = [["a", "b"], ["D"], ["C"]]
    r = M.claim_extraction_f1(pred, gold, _exact_sim)
    # 3 matches (A, B, C); 4 predicted (A, B, X, C); 4 gold (a, b, D, C).
    assert (r["tp"], r["n_pred"], r["n_gold"]) == (3, 4, 4)
    assert r["precision"] == pytest.approx(0.75)
    assert r["recall"] == pytest.approx(0.75)
    assert r["f1"] == pytest.approx(0.75)


def test_claim_extraction_threshold_blocks_weak_matches():
    # Fake similarity 0.5 < 0.8, so "X" never matches "D".
    r = M.claim_extraction_f1([["X"]], [["D"]], _exact_sim)
    assert r["tp"] == 0 and r["f1"] == 0.0


def test_claim_extraction_matches_one_to_one():
    # Two identical predictions can match only one gold claim.
    r = M.claim_extraction_f1([["A", "A"]], [["A"]], _exact_sim)
    assert r["tp"] == 1


def test_recall_at_5():
    retrieved = [["s1", "s2", "s3", "s4", "s5", "own"],   # own at rank 6 -> miss
                 ["x", "own", "y"],                       # hit
                 ["a", "b"]]                              # no gold -> ignored
    gold = [{"own"}, {"own"}, set()]
    assert M.recall_at_k(retrieved, gold, k=5) == pytest.approx(0.5)


# ---------- statistics ----------

def test_bootstrap_identical_models_has_p_one(fake_df):
    r = M.paired_bootstrap(fake_df, fake_df, M.evidence_sensitivity, n_resamples=200)
    assert r["diff"] == 0 and r["p_value"] == pytest.approx(1.0)


def test_bootstrap_detects_a_clear_difference(fake_df):
    better = fake_df.copy()
    better.loc[better["twin_type"] == "B", "pred"] = "contradicted"     # ESR 1.0
    worse = fake_df.copy()
    worse.loc[worse["twin_type"] == "B", "pred"] = "supported"          # ESR 0.0
    r = M.paired_bootstrap(better, worse, M.evidence_sensitivity, n_resamples=200)
    assert r["diff"] == pytest.approx(1.0)
    assert r["ci_low"] > 0 and r["p_value"] < 0.05


def test_bootstrap_is_reproducible_with_seed(fake_df):
    def f1(d):
        o = M.originals(d)
        return M.macro_f1(list(o["gold"]), list(o["pred"]))
    other = fake_df.copy()
    other["pred"] = "supported"
    r1 = M.paired_bootstrap(fake_df, other, f1, n_resamples=100, seed=42)
    r2 = M.paired_bootstrap(fake_df, other, f1, n_resamples=100, seed=42)
    assert r1 == r2


def test_mean_sd():
    assert M.mean_sd([0.70, 0.72, 0.74]) == pytest.approx((0.72, 0.02))
    assert M.mean_sd([0.5]) == (0.5, 0.0)
