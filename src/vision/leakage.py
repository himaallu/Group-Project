"""Authenticity leakage test on VERITE: fused vs decoupled verdict models -> ALR.

Question
    Does a verdict model change its answer when only the photo's "AI look" changes?
    Each ``_V`` twin is the same photo passed through the SD VAE (or SDXL img2img at
    strength 0.15): same content, AI fingerprints. The correct verdict does not change,
    so the authenticity leakage rate (ALR, ``src/eval/metrics.py``) should be 0.

Models (logistic regression on SigLIP 2 features, VERITE train split)
    * ``decoupled`` (our design rule, CLAUDE.md §3): sees only the image-caption match
      features (element-wise product of the L2-normalised SigLIP image and caption
      embeddings; their sum is the cosine). It never sees the authenticity score.
    * ``fused``: sees the image embedding, the caption embedding, the match features
      and the authenticity score ``p_ai`` from the F3 CLIP probe.
    * ``fused_no_auth`` (ablation): ``fused`` without ``p_ai``, to separate the effect
      of the authenticity score from the effect of the raw image embedding.
    C is chosen on the val split (macro-F1 on originals) from 0.01 / 0.1 / 1.
    Logistic regression (lbfgs) is deterministic, so seeds 43/44 give identical runs;
    only seed 42 is reported.

ALR evaluation
    For val + test images with a twin, the original is scored from its JPEG-85 copy and
    the twin from the JPEG-85 twin (both re-saved the same way in F2), so the only
    difference is the generative round trip. VAE and SDXL twins are evaluated
    separately. 95% CIs and fused-vs-decoupled p-values come from the paired bootstrap
    in ``metrics.py`` (1,000 resamples, grouped by original). ``resave_flip`` is the
    noise floor: how often the verdict changes between the raw photo and its JPEG-85
    re-save, with no generative step.

Inputs
    data/processed/vision/verite.jsonl, data/copies/vision/{vae,sdxl}.jsonl
    features/vision/verite/siglip2/... (image + caption embeddings + manifests)
    features/vision/jpeg85_final/siglip2/... ; features/vision/authenticity/clip_probe_p_ai.csv

Outputs
    results/F3/leakage/preds_<model>.jsonl            VERITE test originals (macro-F1, AUROC)
    results/F3/leakage/preds_<model>_<method>.jsonl   originals + _V twins used for ALR
    results/F3/alr.csv, results/F3/alr_bootstrap.json
    (Predictions sit in a subfolder so the team's English/Arabic tables from
    src/eval/report.py, which reads results/*/preds_*.jsonl, do not list them.)

How to run
    python -m src.vision.probe          # first: writes clip_probe_p_ai.csv
    python -m src.vision.leakage
    python -m src.vision.leakage --limit 10   # 10 training records, quick check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.loaders import join_gold, load_jsonl
from src.eval.metrics import auroc, authenticity_leakage, macro_f1, originals, paired_bootstrap
from src.vision.tracking import finish, log_summary, start_run

SIGLIP = Path("features/vision/verite/siglip2")
IMAGE_EMB = SIGLIP / "verified_image_extraction/verite_siglip2_image_projected_embeddings.npy"
IMAGE_MANIFEST = SIGLIP / "verified_image_extraction/verite_siglip2_image_projected_embeddings_manifest.csv"
CAPTION_EMB = SIGLIP / "verified_caption_extraction/verite_siglip2_caption_projected_embeddings.npy"
PAIR_EMB = Path("features/vision/jpeg85_final/siglip2/jpeg85_siglip2_embeddings.npy")
PAIR_MANIFEST = Path("features/vision/jpeg85_final/siglip2/jpeg85_siglip2_embedding_manifest.csv")
P_AI = Path("features/vision/authenticity/clip_probe_p_ai.csv")
TWIN_FILES = {"VAE": "data/copies/vision/vae.jsonl", "SDXL": "data/copies/vision/sdxl.jsonl"}
PAIR_DATASET = {"VAE": "VERITE_VAE", "SDXL": "SDXL"}  # dataset name in the JPEG-85 manifest
MODELS = ("decoupled", "fused", "fused_no_auth")
CLASSES = ("supported", "contradicted")
C_GRID = (0.01, 0.1, 1.0)
SEED = 42


def features(model: str, img: np.ndarray, cap: np.ndarray, p_ai: np.ndarray) -> np.ndarray:
    """Build the input matrix of one model.

    Args:
        model: one of ``MODELS``.
        img, cap: (n, d) L2-normalised SigLIP image / caption embeddings.
        p_ai: (n,) authenticity score of each image.
    """
    match = img * cap  # sums to the image-caption cosine
    if model == "decoupled":
        return match
    parts = [img, cap, match]
    if model == "fused":
        parts.append(p_ai[:, None])
    return np.hstack(parts)


def fit(X: np.ndarray, y: np.ndarray, C: float):
    """Standardise + class-balanced logistic regression."""
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    return make_pipeline(StandardScaler(), clf).fit(X, y)


def to_preds(ids: list[str], clf, X: np.ndarray) -> list[dict]:
    """Predictions in the CLAUDE.md §5 format (``unverified`` gets probability 0)."""
    proba = clf.predict_proba(X)  # columns follow clf.classes_ = [0, 1] -> CLASSES
    preds = []
    for rid, p in zip(ids, proba):
        probs = {"supported": float(p[0]), "contradicted": float(p[1]), "unverified": 0.0}
        preds.append({"id": rid, "pred_label": CLASSES[int(p.argmax())], "probs": probs})
    return preds


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows as JSON Lines."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_verite() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """VERITE records with their image embedding, caption embedding and p_ai (row-aligned)."""
    records = pd.DataFrame(load_jsonl("data/processed/vision/verite.jsonl"))
    image_rows = {name: i for i, name in enumerate(pd.read_csv(IMAGE_MANIFEST)["image_id"])}
    img = np.load(IMAGE_EMB)[records["image_file"].map(image_rows).to_numpy()]
    cap = np.load(CAPTION_EMB)  # row i = record verite_{i:06d} (checked in F2)
    scores = pd.read_csv(P_AI)
    verite_p = scores[scores["dataset"] == "VERITE"].set_index("image_id")["p_ai"]
    p_ai = records["image_file"].str.replace(".jpg", "", regex=False).map(verite_p).to_numpy()
    assert not np.isnan(p_ai).any(), "every VERITE image needs a p_ai score"
    return records, img.astype(np.float32), cap.astype(np.float32), p_ai


def load_pairs(method: str) -> dict[tuple[str, str], tuple[np.ndarray, float]]:
    """(image_id, role) -> (SigLIP embedding, p_ai) for the JPEG-85 originals and twins."""
    meta = pd.read_csv(PAIR_MANIFEST)
    emb = np.load(PAIR_EMB).astype(np.float32)
    scores = pd.read_csv(P_AI)
    scores = scores[scores["dataset"] == PAIR_DATASET[method]].set_index(["image_id", "role"])["p_ai"]
    out = {}
    for i, row in meta[meta["dataset"] == PAIR_DATASET[method]].iterrows():
        stem = row["image_id"].replace(".jpg", "")
        out[(stem, row["role"])] = (emb[row["embedding_row"]], float(scores[(stem, row["role"])]))
    return out


def alr_eval(method: str, clf, model: str, records: pd.DataFrame, cap: np.ndarray) -> list[dict]:
    """Predict each val/test original (JPEG-85 copy) and its ``_V`` twin for one method."""
    pairs = load_pairs(method)
    twins = pd.DataFrame(load_jsonl(TWIN_FILES[method]))
    twins = twins[twins["source"] == "verite"]
    twins = twins[twins["split"].isin(["val", "test"])]
    row_of = {rid: i for i, rid in enumerate(records["id"])}
    ids, img, caps, p_ai = [], [], [], []
    for twin_id, image_file in zip(twins["id"], twins["image_file"]):
        orig_id, stem = twin_id[:-2], image_file.replace(".jpg", "")
        for rid, role in ((orig_id, "original"), (twin_id, "twin")):
            vec, score = pairs[(stem, role)]
            ids.append(rid)
            img.append(vec)
            caps.append(cap[row_of[orig_id]])  # the caption is the same for both
            p_ai.append(score)
    X = features(model, np.array(img), np.array(caps), np.array(p_ai))
    return to_preds(ids, clf, X)


def resave_flip_rate(preds: list[dict], clf, model: str, records: pd.DataFrame,
                     img: np.ndarray, cap: np.ndarray, p_ai: np.ndarray) -> float:
    """Noise floor for ALR: share of originals whose verdict changes between the raw
    photo and its JPEG-85 re-save (no generative step at all).

    Args:
        preds: output of :func:`alr_eval` (its original ids use the JPEG-85 copy).
        clf, model: the fitted model and its name.
        records, img, cap, p_ai: VERITE records and their raw-photo features.
    """
    jpeg_pred = {p["id"]: p["pred_label"] for p in preds if not p["id"].endswith("_V")}
    rows = records.index[records["id"].isin(jpeg_pred)].to_numpy()
    raw = to_preds(list(records["id"].iloc[rows]), clf,
                   features(model, img[rows], cap[rows], p_ai[rows]))
    return float(np.mean([p["pred_label"] != jpeg_pred[p["id"]] for p in raw]))


def main() -> None:
    """Train the three models, predict VERITE test + twins, compute ALR, save."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="N training records (quick check)")
    ap.add_argument("--out", default="results/F3")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    preds_dir = out / "leakage"
    preds_dir.mkdir(parents=True, exist_ok=True)

    records, img, cap, p_ai = load_verite()
    y = (records["label"] == "contradicted").to_numpy().astype(int)
    split = records["split"].to_numpy()
    train = np.where(split == "train")[0][: args.limit]
    gold = load_jsonl("data/processed/vision/verite.jsonl")
    for path in TWIN_FILES.values():
        gold += load_jsonl(path)

    rows, joined = [], {}
    for model in MODELS:
        X = features(model, img, cap, p_ai)
        # Pick C on the val originals, then refit on train (val is never trained on).
        val = split == "val"
        best_C = max(C_GRID, key=lambda C: macro_f1(
            list(records["label"][val]),
            [CLASSES[k] for k in fit(X[train], y[train], C).predict(X[val])]))
        clf = fit(X[train], y[train], best_C)

        test = np.where(split == "test")[0]
        test_preds = to_preds(list(records["id"].iloc[test]), clf, X[test])
        write_jsonl(test_preds, preds_dir / f"preds_{model}.jsonl")
        df_test, _ = join_gold(test_preds, gold)
        orig = originals(df_test)
        probs = orig[["p_supported", "p_contradicted", "p_unverified"]].to_numpy()
        f1, roc = macro_f1(list(orig["gold"]), list(orig["pred"])), auroc(orig["gold"], probs)

        run = start_run(f"F3-leakage-{model}-s{SEED}", {"seed": SEED, "model": model, "C": best_C,
                        "features": "SigLIP 2 so400m"}, not args.no_wandb)
        for method in TWIN_FILES:
            preds = alr_eval(method, clf, model, records, cap)
            write_jsonl(preds, preds_dir / f"preds_{model}_{method.lower()}.jsonl")
            df, missing = join_gold(preds, gold)
            assert not missing, f"no gold record for {missing[:3]}"
            joined[(model, method)] = df
            result = {"model": model, "twin_method": method, "C": best_C,
                      "n_twins": int((df["twin_type"] == "V").sum()),
                      "alr": authenticity_leakage(df),
                      "resave_flip": resave_flip_rate(preds, clf, model, records, img, cap, p_ai),
                      "test_macro_f1": f1, "test_auroc": roc}
            rows.append(result)
            log_summary(run, {f"alr_{method}": result["alr"], "test_macro_f1": f1, "test_auroc": roc})
        finish(run)
        print(f"{model:14s} C={best_C:<5} test macro-F1 {f1:.3f} AUROC {roc:.3f}")

    # 95% CI of each ALR, and a paired test of each fused model against decoupled.
    boot = {}
    for row in rows:
        model, method = row["model"], row["twin_method"]
        row["alr_ci_low"], row["alr_ci_high"] = _ci(joined[(model, method)])
        if model != "decoupled":
            boot[f"{model}_vs_decoupled_{method}"] = paired_bootstrap(
                joined[(model, method)], joined[("decoupled", method)], authenticity_leakage)
    table = pd.DataFrame(rows)
    table.to_csv(out / "alr.csv", index=False)
    (out / "alr_bootstrap.json").write_text(json.dumps(boot, indent=2))
    print(table[["model", "twin_method", "n_twins", "alr", "alr_ci_low", "alr_ci_high",
                 "resave_flip"]].to_string(index=False))


def _ci(df: pd.DataFrame, n_resamples: int = 1000) -> tuple[float, float]:
    """95% bootstrap interval of ALR, resampling originals together with their twins."""
    groups = list(df.groupby("base_id").indices.values())
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(n_resamples):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        value = authenticity_leakage(df.iloc[rows])
        if value is not None:
            values.append(value)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


if __name__ == "__main__":
    main()
