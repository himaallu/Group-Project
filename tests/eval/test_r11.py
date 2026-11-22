"""Unit tests for the R11 tables in src/eval/figures.py (baseline audit + vision).

Run: python -m pytest tests/eval/test_r11.py -q
"""

import pandas as pd

from src.eval.figures import (audit_table, detector_table, leakage_table, onnx_table,
                              robustness_table)


def test_audit_table_formats_seeds_and_drops_empty_columns():
    audit = pd.DataFrame([
        {"model": "claim_only", "dataset": "multifc", "n_seeds": 1, "macro_f1": 0.4805,
         "macro_f1_sd": None, "auroc": 0.719, "auroc_sd": None, "esr": 0.4448, "esr_sd": None,
         "unrelated_rate": 0.012, "unrelated_rate_sd": None, "paraphrase_consistency": 0.845,
         "paraphrase_consistency_sd": None, "ece": 0.11, "ece_sd": None},
        {"model": "similarity", "dataset": "multifc", "n_seeds": 3, "macro_f1": 0.3855,
         "macro_f1_sd": 0.005, "auroc": 0.61, "auroc_sd": 0.002, "esr": 0.59, "esr_sd": 0.01,
         "unrelated_rate": 0.014, "unrelated_rate_sd": 0.002, "paraphrase_consistency": 0.73,
         "paraphrase_consistency_sd": 0.025, "ece": 0.06, "ece_sd": 0.003},
        {"model": "avt_claim_only", "dataset": "averitec", "n_seeds": 1, "macro_f1": 0.3333,
         "macro_f1_sd": None, "auroc": 0.52, "auroc_sd": None, "esr": None, "esr_sd": None,
         "unrelated_rate": None, "unrelated_rate_sd": None, "paraphrase_consistency": None,
         "paraphrase_consistency_sd": None, "ece": 0.16, "ece_sd": None}])
    mf = audit_table(audit, "multifc")
    assert list(mf["baseline"]) == ["claim-only", "similarity + LightGBM"]
    assert mf.loc[0, "macro-F1"] == "48.1" and mf.loc[1, "macro-F1"] == "38.6 ± 0.5"
    avt = audit_table(audit, "averitec")
    assert list(avt.columns) == ["baseline", "seeds", "macro-F1", "AUROC", "ECE"]


def test_vision_tables():
    probe = pd.DataFrame([{"Experiment": "GenImage Authenticity Probe", "Model": "Linear SVM",
                           "Accuracy": 0.507, "Precision": 1.0, "Recall": 0.014, "F1": 0.028,
                           "ROC_AUC": 0.903},
                          {"Experiment": "VERITE Similarity Baseline", "Model": "Decoupled",
                           "Accuracy": 0.57, "Precision": None, "Recall": None, "F1": None,
                           "ROC_AUC": None}])
    cnn = pd.DataFrame([{"model": m, "split": s, "corruption": c, "accuracy": 0.5,
                         "precision": 0.9, "recall": 0.05, "f1": 0.1, "auroc": 0.8}
                        for m in ("resnet50", "vit_b16") for s in ("validation", "test")
                        for c in ("clean", "jpeg50")])
    det = detector_table(probe, cnn)
    assert list(det["model"]) == ["CLIP ViT-L/14 + Linear SVM (frozen)",
                                  "ResNet-50 (last 4 blocks)", "ViT-B/16 (last 4 blocks)"]
    assert len(robustness_table(cnn)) == 4               # 2 models x 2 corruptions, test only
    pairs = pd.DataFrame([{"Split": "test", "Model": "fused", "Pairs": 51,
                           "Original Accuracy": 0.6078, "Twin Accuracy": 0.6078,
                           "Changed Verdicts": 4, "ALR (%)": 7.8431}])
    assert leakage_table(pairs).loc[0, "ALR"] == "7.8"
    lat = pd.DataFrame([{"arch": "resnet50", "onnx_mb": 94.0, "max_abs_diff": 5.96e-07,
                         "ort_cpu_ms_mean": 15.5097, "torch_gpu_ms_mean": 6.6619, "gpu": "NVIDIA L4"}])
    row = onnx_table(lat).iloc[0]
    assert row["ONNX MB"] == "94.0" and row["CPU ms (ONNX Runtime)"] == "15.51"


def test_onnx_int8_table():
    from src.eval.figures import onnx_int8_table

    q = pd.DataFrame([{"arch": "vit_b16", "precision": "int8", "onnx_mb": 87.3, "max_abs_diff": 0.2,
                       "ort_cpu_ms_mean": 432.678, "ort_cpu_ms_p50": 398.8, "test_images": 500,
                       "test_accuracy": 0.576, "test_auroc": 0.77, "same_decision_as_pytorch": 0.982}])
    row = onnx_int8_table(q).iloc[0]
    assert row["precision"] == "INT8" and row["ONNX MB"] == "87.3" and row["CPU ms (ONNX Runtime)"] == "432.68"
