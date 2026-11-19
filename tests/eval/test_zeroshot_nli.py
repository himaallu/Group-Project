"""Tests for the zero-shot NLI baseline rule (F4).

Run: python -m pytest tests/eval/test_zeroshot_nli.py -q
"""
import numpy as np
import pandas as pd

from src.eval.baselines.zeroshot_nli import rule_proba


def test_zeroshot_rule_abstains_below_threshold():
    recs = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    scores = pd.DataFrame({"p_entail": [0.9, 0.1, 0.2], "p_contradict": [0.05, 0.8, 0.1]},
                          index=["a", "b", "c"])
    pred = rule_proba(recs, scores, t=0.5).argmax(1)
    assert list(pred) == [0, 1, 2]   # supported, contradicted, unverified
    assert np.all(rule_proba(recs, scores, 0.5) >= 0)
