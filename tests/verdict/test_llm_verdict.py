"""Tests for src/verdict/llm_verdict.py parts that need no model download."""

from collections import Counter

import pytest

from src.verdict import llm_verdict as L


class FakeTok:
    """Character-level stand-in for a tokenizer (ids = code points)."""
    eos_token = "#"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return "".join(m["content"] for m in messages) + "|A:"


def _rec(rid, label, n=4):
    return {"id": rid, "claim": f"claim {rid}", "label": label,
            "evidence": [{"text": f"snippet {i}"} for i in range(n)]}


def test_top_snippets_by_relevance_and_twin_fallback():
    rel = {"m_1": {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.2}}
    assert L.top_snippets(_rec("m_1", "supported"), rel) == ["snippet 1", "snippet 2", "snippet 3"]
    assert L.top_snippets(_rec("m_1_B", "contradicted"), rel)[0] == "snippet 1"   # source scores
    assert L.top_snippets(_rec("m_9_C", "unverified"), rel) == ["snippet 0", "snippet 1", "snippet 2"]


def test_sample_training_balanced_and_twins_added():
    origs = [_rec(f"m_{i}", "supported") for i in range(10)] + \
            [_rec(f"m_{i}", "contradicted") for i in range(10, 20)] + [_rec("m_99", "unverified")]
    twins = [_rec(f"m_{i}_{k}", "x") for i in range(5) for k in "PBC"]
    orig_only = L.sample_training(origs, twins, "orig", n_per_class=4, n_twin_per_type=2)
    assert Counter(r["label"] for r in orig_only) == {"supported": 4, "contradicted": 4, "unverified": 4}
    with_twins = L.sample_training(origs, twins, "twins", n_per_class=4, n_twin_per_type=2)
    assert len(with_twins) == 12 + 6
    assert L.sample_training(origs, twins, "orig", 4, 2) == orig_only        # seeded


def test_encode_example_masks_prompt_and_keeps_answer():
    ex = L.encode_example(FakeTok(), _rec("m_1", "supported"), {}, max_len=512)
    answer = [ord(c) for c in "supported#"]
    assert ex["input_ids"][-len(answer):] == answer
    assert ex["labels"][-len(answer):] == answer
    assert all(v == -100 for v in ex["labels"][:-len(answer)])


def test_encode_example_truncates_prompt_from_the_left():
    ex = L.encode_example(FakeTok(), _rec("m_1", "unverified", n=3), {}, max_len=40)
    assert len(ex["input_ids"]) == 40 and ex["input_ids"][-1] == ord("#")


def test_label_tokens_must_differ():
    class SameStart(FakeTok):
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2]}
    assert len(set(L.label_token_ids(FakeTok()))) == 3
    with pytest.raises(ValueError):
        L.label_token_ids(SameStart())
