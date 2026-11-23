"""Extra checks on Feba's vision results, computed from files already in results/ (H6).

What it does
    No model is rerun; everything comes from pushed prediction files.
    1. Leakage significance (VERITE): exact McNemar test on the pairs where the decoupled
       and fused models disagree about whether the image twin changed the verdict, on the
       test split and on validation + test pooled.
    2. Why verdicts flip: for every flipped pair, the change in authenticity score from
       original to twin and the original caption-score gap |p_true - p_miscaptioned|.
    3. CNN vs ViT on the unseen generator (GenImage test, Stable Diffusion v1.4): AUROC of
       ResNet-50 and ViT-B/16 with paired bootstrap 95% CIs (1,000 resamples, seed 42).
    4. ResNet-50 on validation: accuracy at cut-offs 0.5 / 0.2 / 0.1, score distribution
       (Feba's RCA point 4). Cut-offs are only explored on validation, never chosen on test.

Inputs
    results/F3/F3_pair_verdicts.csv, results/F3/F3_verite_authenticity_scores.csv,
    results/F6/predictions.csv

Outputs
    results/H6/vision_checks/vision_checks.json

How to run
    python -m src.eval.vision_checks
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

F3 = Path("results/F3")
OUT = Path("results/H6/vision_checks")


def flips(verdicts: pd.DataFrame, model: str) -> pd.Series:
    """True for each (image, split) whose verdict differs between original and twin."""
    v = verdicts[verdicts.model == model].pivot_table(
        index=["image_file", "split"], columns="version", values="picks_true_caption", aggfunc="first")
    return v["original"] != v["twin"]


def mcnemar(a: pd.Series, b: pd.Series) -> dict:
    """Exact McNemar test: pairs changed only by b vs only by a."""
    only_b, only_a = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(only_b, only_b + only_a, 0.5).pvalue if only_a + only_b else 1.0
    return {"pairs": int(len(a)), "changed_a": int(a.sum()), "changed_b": int(b.sum()),
            "only_b": only_b, "only_a": only_a, "p_exact": round(float(p), 4)}


def leakage(verdicts: pd.DataFrame, auth: pd.DataFrame) -> dict:
    """Checks 1 and 2: McNemar decoupled vs fused, and what the flipped pairs look like."""
    dec, fus = flips(verdicts, "decoupled"), flips(verdicts, "fused")
    out = {}
    for name, splits in (("test", ["test"]), ("validation_plus_test", ["validation", "test"])):
        keep = dec.index.get_level_values("split").isin(splits)
        out[f"mcnemar_{name}"] = mcnemar(dec[keep], fus[keep])
    # Caption-score gap of the original pair, per model, and authenticity shift per photo.
    verdicts = verdicts.assign(gap=(verdicts.p_true_caption - verdicts.p_miscaptioned).abs())
    shift = (auth.set_index("image_file").twin_authenticity_score
             - auth.set_index("image_file").original_authenticity_score)
    out["authenticity_shift_all_photos"] = {
        "n": int(len(shift)), "median": round(float(shift.median()), 4),
        "mean": round(float(shift.mean()), 4), "abs_over_0.1": int((shift.abs() > 0.1).sum()),
        "abs_over_0.3": int((shift.abs() > 0.3).sum())}
    for model, changed in (("decoupled", dec), ("fused", fus)):
        test = changed[changed.index.get_level_values("split") == "test"]
        files = [f for f, _ in test[test].index]
        gaps = verdicts[(verdicts.model == model) & (verdicts.version == "original")
                        & (verdicts.split == "test")].set_index("image_file").gap
        out[f"test_flips_{model}"] = {
            "n": len(files),
            "authenticity_shift": [round(float(shift[f]), 3) for f in files],
            "caption_gap": [round(float(gaps[f]), 4) for f in files],
            "median_caption_gap_all_test": round(float(gaps.median()), 4)}
    # Photos whose verdict only the fused model changed: is their authenticity shift unusual?
    # All splits count here: no model saw any twin in training.
    for name, splits in (("all_splits", ["train", "validation", "test"]),
                         ("validation_plus_test", ["validation", "test"])):
        only = fus & ~dec
        only = only[only.index.get_level_values("split").isin(splits)]
        files = sorted(f for f, _ in only[only].index)
        s = shift[files]
        out[f"fused_only_flips_{name}"] = {
            "n": len(files), "abs_shift_over_0.1": int((s.abs() > 0.1).sum()),
            "share_abs_shift_over_0.1": round(float((s.abs() > 0.1).mean()), 3) if files else None,
            "share_all_photos": round(float((shift.abs() > 0.1).mean()), 3),
            "shift": {f: round(float(shift[f]), 3) for f in files}}
    return out


def detectors(pred: pd.DataFrame) -> dict:
    """Checks 3 and 4: AUROC with bootstrap CIs on test; ResNet-50 cut-offs on validation."""
    out = {}
    test = pred[pred.split == "test"]
    y = (test.class_name != "real").astype(int).values
    r, v = test.p_ai_resnet50.values, test.p_ai_vit_b16.values
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        boot.append((roc_auc_score(y[i], r[i]), roc_auc_score(y[i], v[i])))
    boot = np.array(boot)
    ci = lambda x: [round(float(q), 3) for q in np.percentile(x, [2.5, 97.5])]
    out["test_auroc"] = {"resnet50": round(roc_auc_score(y, r), 3), "resnet50_ci": ci(boot[:, 0]),
                         "vit_b16": round(roc_auc_score(y, v), 3), "vit_b16_ci": ci(boot[:, 1]),
                         "diff_vit_minus_resnet": round(float(np.mean(boot[:, 1] - boot[:, 0])), 3),
                         "diff_ci": ci(boot[:, 1] - boot[:, 0]),
                         "p_one_sided": float(np.mean(boot[:, 1] - boot[:, 0] <= 0))}
    val = pred[pred.split == "validation"]
    yv, s = (val.class_name != "real").astype(int).values, val.p_ai_resnet50.values
    out["resnet50_validation"] = {
        "auroc": round(roc_auc_score(yv, s), 3),
        "biggan_median_score": round(float(np.median(s[yv == 1])), 3),
        "biggan_missed_at_0.5": int((s[yv == 1] < 0.5).sum()),
        "missed_between_0.2_and_0.5": int(((s[yv == 1] >= 0.2) & (s[yv == 1] < 0.5)).sum()),
        "real_99th_percentile": round(float(np.percentile(s[yv == 0], 99)), 3),
        "accuracy_by_cutoff": {str(t): round(float(((s >= t) == yv).mean()), 3) for t in (0.5, 0.2, 0.1)}}
    return out


def main() -> None:
    """Run all checks and write one JSON file."""
    verdicts = pd.read_csv(F3 / "F3_pair_verdicts.csv")
    auth = pd.read_csv(F3 / "F3_verite_authenticity_scores.csv")
    pred = pd.read_csv("results/F6/predictions.csv")
    result = {"leakage": leakage(verdicts, auth), "detectors": detectors(pred)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vision_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
