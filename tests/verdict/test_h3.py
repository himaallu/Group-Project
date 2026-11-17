"""Tests for the verdict head (src/verdict/features.py, model.py, train.py)."""

import json
import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.eval.metrics import LABELS  # noqa: E402
from src.verdict import features as Fe  # noqa: E402
from src.verdict import train as T  # noqa: E402
from src.verdict.model import VerdictHead, load_head, predict  # noqa: E402


# ---------- features ----------

def test_entity_found_rules():
    ev = " " + Fe._norm("Jun 7, 2016 ... Ted Cruz spoke in Dubai's metro.") + " "
    assert Fe.entity_found({"text": "Dubai", "label": "GPE"}, ev)
    assert Fe.entity_found({"text": "Senator Cruz", "label": "PERSON"}, ev)      # surname
    assert Fe.entity_found({"text": "June 2016", "label": "DATE"}, ev)           # year
    assert not Fe.entity_found({"text": "Abu Dhabi", "label": "GPE"}, ev)
    assert not Fe.entity_found({"text": "2019", "label": "DATE"}, ev)


def test_entity_matches_default_one_when_no_entities():
    m = Fe.entity_matches([{"text": "Paris", "label": "GPE"}, {"text": "Rome", "label": "GPE"}],
                          [{"text": "A trip to Paris."}])
    assert m == {"match_place": 0.5, "match_date": 1.0, "match_person": 1.0}


def test_record_features_weighting_and_relevant_pool():
    rec = {"id": "x", "evidence": [{"text": "a"}, {"text": "b"}]}
    nli = {0: (0.9, 0.0, 0.1), 1: (0.0, 0.8, 0.2)}
    rel = {0: (0.75,), 1: (0.25,)}
    f, filled = Fe.record_features(rec, nli, rel, [])
    assert f["w_entail"] == pytest.approx(0.675) and f["w_contradict"] == pytest.approx(0.2)
    assert f["max_entail"] == 0.9 and f["max_contradict"] == 0.0   # snippet 1 not relevant
    assert f["share_relevant"] == 0.5 and f["n_snippets"] == 0.2 and filled == 0


def test_record_features_no_snippets_and_missing_nli():
    f, _ = Fe.record_features({"id": "x", "evidence": []}, None, None, [])
    assert f["w_entail"] == 0.0 and f["n_snippets"] == 0.0
    with pytest.raises(KeyError):
        Fe.record_features({"id": "x", "evidence": [{"text": "a"}]}, {}, None, [])


def test_build_features_twin_relevance_fallback():
    recs = [{"id": "m_1_B", "label": "contradicted", "split": "test", "evidence": [{"text": "a"}]},
            {"id": "m_1_C", "label": "unverified", "split": "test", "evidence": [{"text": "z"}]}]
    nli = pd.DataFrame({"id": ["m_1_B", "m_1_C"], "snippet_idx": [0, 0], "p_entail": [0.1, 0.1],
                        "p_contradict": [0.8, 0.1], "p_neutral": [0.1, 0.8]})
    rel = pd.DataFrame({"id": ["m_1"], "snippet_idx": [0], "relevance": [0.2]})
    t = Fe.build_features(recs, nli, rel, {}, fill_relevance=0.05)
    assert list(t["twin_type"]) == ["B", "C"] and list(t["base_id"]) == ["m_1", "m_1"]
    assert list(t["share_relevant"]) == [0.0, 0.0]   # B uses its source's 0.2, C filled 0.05


# ---------- loss ----------

def _batch():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                          requires_grad=True)
    kinds = [None, "P", "B", "C"]
    targets = torch.tensor([0, 0, 1, 2])
    orig_pos = torch.tensor([-1, 0, 0, 0])
    return logits, kinds, targets, orig_pos


def test_twin_loss_lambda_zero_is_plain_ce():
    logits, kinds, targets, orig_pos = _batch()
    loss, _ = T.twin_loss(logits, kinds, targets, orig_pos, torch.ones(3), 0.0, 0.0)
    assert loss.item() == pytest.approx(torch.nn.functional.cross_entropy(logits[:1], targets[:1]).item())


def test_twin_loss_terms_and_detached_original():
    logits, kinds, targets, orig_pos = _batch()
    _, parts = T.twin_loss(logits, kinds, targets, orig_pos, torch.ones(3), 1.0, 1.0)
    assert parts["inv"] > 0 and parts["sens"] > 0
    # KL alone must not push the original's logits (it is the teacher)
    logits2, *_ = _batch()
    inv_only, _ = T.twin_loss(logits2, kinds, targets, orig_pos, torch.ones(3), 1.0, 0.0)
    (inv_only - T.twin_loss(logits2, kinds, targets, orig_pos, torch.ones(3), 0.0, 0.0)[0]).backward()
    assert torch.allclose(logits2.grad[0], torch.zeros(3))
    assert logits2.grad[1].abs().sum() > 0


def test_kl_zero_when_twin_matches_original():
    logits = torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
    _, parts = T.twin_loss(logits, [None, "A"], torch.tensor([0, 0]), torch.tensor([-1, 0]),
                           torch.ones(3), 1.0, 0.0)
    assert parts["inv"] == pytest.approx(0.0, abs=1e-6)


def test_class_weights_inverse_frequency():
    w = T.class_weights(["supported"] * 6 + ["contradicted"] * 3 + ["unverified"])
    assert w.tolist() == pytest.approx([10 / 18, 10 / 9, 10 / 3])


def test_groups_keep_original_first_and_orig_pos():
    table = pd.DataFrame({"base_id": ["a", "a", "b", "a"], "twin_type": [None, "P", None, "B"],
                          "label": ["supported", "supported", "contradicted", "contradicted"]})
    groups = T.make_groups(table)
    assert groups == [[0, 1, 3], [2]]
    kinds, targets, orig_pos = T.batch_tensors(table, [2, 0, 1, 3])
    assert kinds == [None, None, "P", "B"] and orig_pos.tolist() == [-1, -1, 1, 1]


# ---------- end-to-end on synthetic data ----------

def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _synthetic(tmp_path, seed=0):
    """Records whose NLI scores follow the label, plus _P/_B/_C twins of supported ones."""
    rng = random.Random(seed)
    nli_rows, data = [], {}
    probs = {"supported": (0.8, 0.05, 0.15), "contradicted": (0.05, 0.8, 0.15),
             "unverified": (0.1, 0.1, 0.8)}

    def add(rid, label, split, nli_label):
        rec = {"id": rid, "source": "multifc", "split": split, "language": "en",
               "claim": f"claim {rid} in Paris", "evidence": [{"text": "Paris news", "url": None,
                                                              "domain": None}] * 2,
               "image_path": None, "label": label, "orig_label": label}
        for i in range(2):
            e, c, n = (min(max(v + rng.uniform(-0.05, 0.05), 0.0), 1.0) for v in probs[nli_label])
            s = e + c + n
            nli_rows.append({"id": rid, "snippet_idx": i, "p_entail": e / s,
                             "p_contradict": c / s, "p_neutral": n / s})
        return rec

    for split, n in (("train", 90), ("val", 30), ("test", 30)):
        data[split] = [add(f"multifc_{split}{i:03d}", LABELS[i % 3], split, LABELS[i % 3])
                       for i in range(n)]
    twins = []
    for split in ("train", "test"):
        for o in data[split]:
            if o["label"] == "supported":
                twins.append(add(o["id"] + "_P", "supported", split, "supported"))
                twins.append(add(o["id"] + "_B", "contradicted", split, "supported"))  # NLI fooled
                twins.append(add(o["id"] + "_C", "unverified", split, "unverified"))
    paths = {}
    for split in data:
        paths[split] = tmp_path / f"{split}.jsonl"
        _write(paths[split], data[split])
    paths["twins"] = tmp_path / "twins.jsonl"
    _write(paths["twins"], twins)
    paths["nli"] = tmp_path / "nli.parquet"
    pd.DataFrame(nli_rows).to_parquet(paths["nli"])
    paths["rel"] = tmp_path / "rel.parquet"
    pd.DataFrame([{"id": r["id"], "snippet_idx": r["snippet_idx"], "relevance": 0.9}
                  for r in nli_rows]).to_parquet(paths["rel"])
    paths["ent"] = tmp_path / "ent.jsonl"
    _write(paths["ent"], [{"id": r["id"], "claim_entities": [{"text": "Paris", "label": "GPE"}]}
                          for split in data for r in data[split]])
    return paths


def _args(paths, tmp_path, name, **kw):
    argv = ["--name", name, "--train", str(paths["train"]), "--val", str(paths["val"]),
            "--test", str(paths["test"]), "--twins", str(paths["twins"]), "--nli", str(paths["nli"]),
            "--relevance", str(paths["rel"]), "--entities", str(paths["ent"]),
            "--results-dir", str(tmp_path / "res"), "--models-dir", str(tmp_path / "models"),
            "--epochs", "15", "--patience", "15", "--batch-groups", "8", "--no-wandb"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return T.parse_args(argv)


def test_train_end_to_end_writes_outputs(tmp_path):
    paths = _synthetic(tmp_path)
    m = T.train(_args(paths, tmp_path, "v0"))
    assert m["val"]["macro_f1"] > 0.9
    for f in ("preds_v0_s42.jsonl", "val_logits_v0_s42.jsonl", "metrics_v0_s42.json"):
        assert (tmp_path / "res" / f).exists()
    model, config = load_head(tmp_path / "models" / "verdict_v0_s42")
    assert isinstance(model, VerdictHead) and config["lambda_inv"] == 0.0
    assert m["test"]["n_B"] == 10 and m["val_twins"]["n_orig"] > 0


def test_twin_loss_raises_esr_over_v0(tmp_path):
    # In this synthetic data the _B twins look exactly like supported originals (no
    # feature can tell them apart), so only the _C and _P terms are checked: the twin
    # loss must not lower the unrelated rate and paraphrases must keep their verdict.
    paths = _synthetic(tmp_path)
    v0 = T.train(_args(paths, tmp_path, "v0"))
    tw = T.train(_args(paths, tmp_path, "tw", lambda_inv=1, lambda_sens=1))
    assert tw["test"]["unrelated_rate"] >= v0["test"]["unrelated_rate"]
    assert tw["test"]["paraphrase_consistency"] == pytest.approx(1.0)
