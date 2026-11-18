"""End-to-end test of src/eval/run_h5.py and src/verdict/predict.py on synthetic data."""

import json
import shutil

import pandas as pd
import pytest

pytest.importorskip("torch")

from src.eval import run_h5  # noqa: E402
from src.verdict.predict import predict_records  # noqa: E402
from tests.verdict.test_h3 import _synthetic  # noqa: E402


def _layout(tmp_path):
    """Put the synthetic files where run_h5 expects them."""
    p = _synthetic(tmp_path)
    en = tmp_path / "en"
    en.mkdir()
    for split in ("train", "val", "test"):
        shutil.copy(p[split], en / f"multifc_{split}.jsonl")
    shutil.copy(p["test"], en / "averitec_test.jsonl")      # stands in for AVeriTeC
    nli_dir = tmp_path / "nli"
    (nli_dir / "copies_en").mkdir(parents=True)
    for name in ("multifc_train", "multifc_val", "multifc_test", "averitec_test",
                 "copies_en/train", "copies_en/test"):
        shutil.copy(p["nli"], nli_dir / f"{name}.parquet")
    sel = tmp_path / "selected.json"
    sel.write_text(json.dumps({"chosen": {"lambda_inv": 1.0, "lambda_sens": 0.5}}))
    return p, en, nli_dir, sel


def test_run_h5_end_to_end(tmp_path):
    p, en, nli_dir, sel = _layout(tmp_path)
    out = tmp_path / "H5"
    table = run_h5.main([
        "--nli-dir", str(nli_dir), "--out-dir", str(out), "--models-dir", str(tmp_path / "m"),
        "--seeds", "42", "--selected", str(sel), "--en-dir", str(en),
        "--twins", str(p["twins"]), "--relevance", str(p["rel"]), "--entities", str(p["ent"]),
        "--averitec", str(en / "averitec_test.jsonl"), "--epochs", "4", "--no-wandb"])
    expected = {"twin", "notwin", "twin_norel", "twin_noent", "twin_cal", "notwin_cal",
                "avt_twin", "avt_notwin", "avt_twin_norel", "avt_twin_noent",
                "avt_twin_cal", "avt_notwin_cal"}
    assert expected <= set(table.index)
    boot = json.loads((out / "bootstrap.json").read_text())
    assert "twin vs twin_norel (s42)" in boot and "avt_twin vs avt_notwin (s42)" in boot
    assert (out / "summary.csv").exists()
    assert json.loads((tmp_path / "m" / "verdict_twin_norel_s42" / "config.json").read_text())["no_relevance"]


def test_run_h5_missing_nli_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing NLI files"):
        run_h5.nli_files(tmp_path)


def test_predict_matches_training_predictions(tmp_path):
    """predict.py on the test file reproduces the predictions written during training."""
    from src.eval.loaders import load_jsonl
    from src.verdict import train as T
    from src.verdict.features import load_entities
    p = _synthetic(tmp_path)
    args = T.parse_args(["--name", "x", "--train", str(p["train"]), "--val", str(p["val"]),
                         "--test", str(p["test"]), "--nli", str(p["nli"]),
                         "--relevance", str(p["rel"]), "--entities", str(p["ent"]),
                         "--results-dir", str(tmp_path / "r"), "--models-dir", str(tmp_path / "m"),
                         "--epochs", "3", "--no-wandb"])
    T.train(args)
    rows = predict_records(tmp_path / "m" / "verdict_x_s42", load_jsonl(p["test"]),
                           pd.read_parquet(p["nli"]), pd.read_parquet(p["rel"]),
                           load_entities(p["ent"]))
    trained = [json.loads(l) for l in open(tmp_path / "r" / "preds_x_s42.jsonl")]
    assert rows == trained


def test_nli_files_accepts_flat_twin_layout(tmp_path):
    """Rathi's score.py writes twin scores as <dir>/{train,test}.parquet (no copies_en/)."""
    for name in ("multifc_train", "multifc_val", "multifc_test", "averitec_test", "train", "test"):
        (tmp_path / f"{name}.parquet").write_bytes(b"")
    files = run_h5.nli_files(tmp_path)
    assert files["train"][-2:] == [str(tmp_path / "train.parquet"), str(tmp_path / "test.parquet")]


def test_e5_probs_to_logits_round_trip():
    """log-probabilities are valid logits: softmax gives the probabilities back."""
    import numpy as np
    from src.eval.run_e5_ar import probs_to_logits
    from src.verdict.calibrate import softmax
    rows = [{"probs": {"supported": 0.2, "contradicted": 0.7, "unverified": 0.1}}]
    assert np.allclose(softmax(probs_to_logits(rows)), [[0.2, 0.7, 0.1]], atol=1e-6)
