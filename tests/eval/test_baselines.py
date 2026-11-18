"""Tests for the claim-only and similarity baselines on 10 fake records (no model downloads).

Run: python -m pytest tests/eval/test_baselines.py -q
"""
from src.eval.baselines.claim_only import fit_claim_only, predict
from src.eval.baselines.similarity import fit_vectorizer, record_features


def fake_records():
    """10 records: supported claims share words with their evidence."""
    recs = []
    for i in range(10):
        label = ["supported", "contradicted", "unverified"][i % 3]
        claim = f"city {i} opened a new bridge" if label == "supported" else f"aliens landed in town {i}"
        recs.append({"id": f"multifc_{i:06d}", "label": label, "claim": claim,
                     "evidence": [{"text": f"city {i} opened the bridge today", "url": "", "domain": ""}]})
    return recs


def test_claim_only_ignores_evidence():
    recs = fake_records()
    model, C, _ = fit_claim_only(recs, recs)
    twin = dict(recs[0], id="multifc_000000_C", evidence=[])  # same claim, evidence removed
    a, b = predict(model, [recs[0], twin])
    assert a["probs"] == b["probs"]                            # claim-only cannot react to evidence
    assert set(a["probs"]) == {"supported", "contradicted", "unverified"}


def test_similarity_features():
    recs = fake_records()
    vec = fit_vectorizer(recs)
    feats = record_features(recs[0], vec)
    assert len(feats) == 12 and feats[0] == 1 and feats[2] > 0.3   # one snippet, high cosine
    assert record_features(dict(recs[0], evidence=[]), vec)[-1] == 0  # has_evidence = 0

