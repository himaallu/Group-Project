"""Authenticity probe: logistic regression on CLIP ViT-L/14 features (UnivFD-style).

What it does
    * Trains a camera-vs-AI logistic probe on the cached GenImage CLIP embeddings
      (F2). The regularisation strength C is picked by 5-fold cross-validation on the
      training images only (grid 0.1 / 1 / 10, scoring AUROC).
    * Three settings (GenImage has train + validation only here; the Midjourney test
      images were not available, so the validation split is the held-out test and
      no setting tunes anything on it):
        - ``seen``           train real + SD v1.4 + BigGAN -> test real + SD v1.4 + BigGAN
        - ``unseen_biggan``  train real + SD v1.4          -> test real + BigGAN
        - ``unseen_sd``      train real + BigGAN           -> test real + SD v1.4
    * If ``features/vision/genimage/jpeg85_control_clip.npy`` exists (made by
      ``src/vision/jpeg_control.py``), also tests the ``seen`` probe on validation
      images re-saved as JPEG quality 85. In GenImage every fake is PNG and every real
      is JPEG, so this shows how much the probe relies on the file format.
    * Scores every VERITE, M4FC and JPEG-85 original/twin image with the ``seen`` probe
      (``p_ai`` = probability of AI-generated) and compares originals with their twins.
    * The "edited" class is not trained: no edited images are in our data (limitation).

Inputs
    features/vision/genimage/genimage_clip_vit_l14_image_projected_embeddings_float16.npy (+ manifest)
    features/vision/{verite,m4fc,jpeg85_final}/.../*clip*.npy (+ manifests)

Outputs
    results/F3/probe_metrics.csv          one row per setting: accuracy, AUROC, per-class accuracy
    results/F3/twin_authenticity.csv      mean p_ai of originals vs twins per twin method
    features/vision/authenticity/clip_probe_p_ai.csv   p_ai for every scored image
    models/F3/clip_probe_seen.npz         probe weights (coef, intercept, C)

How to run
    python -m src.vision.probe            # full run (CPU, about 1 minute)
    python -m src.vision.probe --limit 10 # 10 images per class, quick check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score

from src.vision.tracking import finish, log_summary, start_run

GENIMAGE = Path("features/vision/genimage")
CLIP_EMB = GENIMAGE / "genimage_clip_vit_l14_image_projected_embeddings_float16.npy"
CLIP_MANIFEST = GENIMAGE / "genimage_image_embedding_manifest.csv"
JPEG_CONTROL_EMB = GENIMAGE / "jpeg85_control_clip.npy"
JPEG_CONTROL_MANIFEST = GENIMAGE / "jpeg85_control_manifest.csv"

# Images scored with the final probe: (dataset name, embeddings, manifest, id column)
SCORED_SETS = [
    ("VERITE", "features/vision/verite/clip_vit_l14/verified_image_extraction/verite_clip_vit_l14_image_projected_embeddings.npy",
     "features/vision/verite/clip_vit_l14/verified_image_extraction/verite_clip_vit_l14_image_projected_embeddings_manifest.csv", "image_id"),
    ("M4FC", "features/vision/m4fc/clip_vit_l14/m4fc_clip_vit_l14_image_embeddings.npy",
     "features/vision/m4fc/clip_vit_l14/m4fc_clip_vit_l14_image_embeddings_manifest.csv", "id"),
    ("JPEG85_PAIRS", "features/vision/jpeg85_final/clip_vit_l14/jpeg85_clip_vit_l14_embeddings.npy",
     "features/vision/jpeg85_final/clip_vit_l14/jpeg85_clip_vit_l14_embedding_manifest.csv", "image_id"),
]

SETTINGS = {
    "seen": (["real", "Stable Diffusion v1.4", "BigGAN"], ["real", "Stable Diffusion v1.4", "BigGAN"]),
    "unseen_biggan": (["real", "Stable Diffusion v1.4"], ["real", "BigGAN"]),
    "unseen_sd": (["real", "BigGAN"], ["real", "Stable Diffusion v1.4"]),
}
C_GRID = [0.1, 1.0, 10.0]
SEED = 42


def load_genimage() -> tuple[np.ndarray, pd.DataFrame]:
    """Load GenImage CLIP embeddings (as float32) and their row manifest."""
    X = np.load(CLIP_EMB).astype(np.float32)
    meta = pd.read_csv(CLIP_MANIFEST)
    assert len(meta) == len(X), "manifest and embeddings must have the same rows"
    return X, meta


def subset(meta: pd.DataFrame, split: str, generators: list[str], limit: int | None) -> np.ndarray:
    """Row positions of one split restricted to some generators (optionally N per generator)."""
    rows = meta[(meta["split"] == split) & meta["generator"].isin(generators)]
    if limit:
        rows = rows.groupby("generator", group_keys=False).head(limit)
    return rows.index.to_numpy()


def train_probe(X: np.ndarray, y: np.ndarray) -> LogisticRegressionCV:
    """Fit the logistic probe; C chosen by 5-fold CV (AUROC) on the training rows only."""
    probe = LogisticRegressionCV(Cs=C_GRID, cv=5, scoring="roc_auc", max_iter=5000,
                                 class_weight="balanced", random_state=SEED)
    return probe.fit(X, y)


def evaluate(probe, X: np.ndarray, meta: pd.DataFrame) -> dict:
    """Accuracy, AUROC and accuracy per generator for the rows in ``meta``.

    Args:
        probe: fitted probe.
        X: embeddings of the rows in ``meta`` (same order).
        meta: rows with ``class_name`` (real / fake) and ``generator``.
    """
    y = (meta["class_name"] == "fake").to_numpy().astype(int)
    p_ai = probe.predict_proba(X)[:, 1]
    pred = (p_ai >= 0.5).astype(int)
    out = {"n": len(y), "accuracy": accuracy_score(y, pred), "auroc": roc_auc_score(y, p_ai)}
    for gen, idx in meta.groupby("generator").indices.items():
        out[f"acc_{gen}"] = accuracy_score(y[idx], pred[idx])
    return out


def score_sets(probe) -> pd.DataFrame:
    """p_ai for every VERITE, M4FC and JPEG-85 original/twin image."""
    frames = []
    for name, emb_path, manifest_path, id_col in SCORED_SETS:
        X = np.load(emb_path).astype(np.float32)
        meta = pd.read_csv(manifest_path)
        assert len(meta) == len(X), f"{manifest_path}: row count differs from embeddings"
        frame = pd.DataFrame({
            "dataset": meta["dataset"] if "dataset" in meta else name,
            "image_id": meta[id_col].astype(str).str.replace(".jpg", "", regex=False),
            "role": meta["role"] if "role" in meta else "plain",
            "p_ai": probe.predict_proba(X)[:, 1],
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def twin_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Compare p_ai of each JPEG-85 original with its twin, per twin method.

    Returns one row per method (VERITE_VAE, M4FC_VAE, SDXL) with mean p_ai of
    originals and twins, the share of each flagged as AI (p_ai >= 0.5) and the share
    of pairs where the twin scores higher than its original.
    """
    pairs = scores[scores["role"].isin(["original", "twin"])]
    wide = pairs.pivot_table(index=["dataset", "image_id"], columns="role", values="p_ai").dropna()
    rows = []
    for method, part in wide.groupby(level="dataset"):
        rows.append({
            "method": method, "n_pairs": len(part),
            "mean_p_ai_original": part["original"].mean(), "mean_p_ai_twin": part["twin"].mean(),
            "flagged_ai_original": (part["original"] >= 0.5).mean(),
            "flagged_ai_twin": (part["twin"] >= 0.5).mean(),
            "twin_higher_share": (part["twin"] > part["original"]).mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Train and evaluate the three settings, score the VERITE/M4FC/twin images, save."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="N images per generator (quick test)")
    ap.add_argument("--out", default="results/F3")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, meta = load_genimage()
    rows, probes = [], {}
    for setting, (train_gens, test_gens) in SETTINGS.items():
        tr = subset(meta, "train", train_gens, args.limit)
        te = subset(meta, "validation", test_gens, args.limit)
        y_tr = (meta.loc[tr, "class_name"] == "fake").to_numpy().astype(int)
        probe = train_probe(X[tr], y_tr)
        probes[setting] = probe
        result = {"setting": setting, "train": "+".join(train_gens), "test": "+".join(test_gens),
                  "C": float(probe.C_[0]), **evaluate(probe, X[te], meta.loc[te])}
        rows.append(result)
        run = start_run(f"F3-probe-{setting}-s{SEED}", {"seed": SEED, "setting": setting,
                        "C_grid": C_GRID, "features": "CLIP ViT-L/14 projected"}, not args.no_wandb)
        log_summary(run, {k: v for k, v in result.items() if isinstance(v, float)})
        finish(run)
        print(f"{setting:14s} AUROC {result['auroc']:.3f}  acc {result['accuracy']:.3f}  C={result['C']}")

    # Same probe on validation images re-saved as JPEG 85 (format control), if available.
    if JPEG_CONTROL_EMB.exists() and not args.limit:
        Xc = np.load(JPEG_CONTROL_EMB).astype(np.float32)
        mc = pd.read_csv(JPEG_CONTROL_MANIFEST)
        for variant, idx in mc.groupby("variant").indices.items():
            result = {"setting": f"seen_on_{variant}", "train": rows[0]["train"], "test": rows[0]["test"],
                      "C": rows[0]["C"], **evaluate(probes["seen"], Xc[idx], mc.iloc[idx].reset_index(drop=True))}
            rows.append(result)
            print(f"seen on {variant:10s} AUROC {result['auroc']:.3f}  acc {result['accuracy']:.3f}")

    pd.DataFrame(rows).to_csv(out / "probe_metrics.csv", index=False)
    if args.limit:
        return

    scores = score_sets(probes["seen"])
    Path("features/vision/authenticity").mkdir(parents=True, exist_ok=True)
    scores.to_csv("features/vision/authenticity/clip_probe_p_ai.csv", index=False)
    summary = twin_summary(scores)
    summary.to_csv(out / "twin_authenticity.csv", index=False)
    print(summary.to_string(index=False))

    Path("models/F3").mkdir(parents=True, exist_ok=True)
    seen = probes["seen"]
    np.savez("models/F3/clip_probe_seen.npz", coef=seen.coef_, intercept=seen.intercept_, C=seen.C_)


if __name__ == "__main__":
    main()
