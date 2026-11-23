from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ("decoupled", "fused", "fused + image embedding")
SIGLIP = {
    "image": "features/vision/verite/siglip2_so400m/embeddings_normalized.npy",
    "caption": "features/vision/verite/siglip2_text/embeddings_normalized.npy",
    "twin": "features/vision/verite/siglip2_vae_twins/embeddings_normalized.npy",
}
CLIP_DIR = "features/vision/verite/clip_vitl14_debiased96"


def build_captions(drive: str | Path, auth: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    drive = Path(drive)
    verite = pd.read_csv("data/processed/vision/verite_manifest.csv")
    twins = pd.read_csv("data/copies/vision/verite_vae_twins.csv")
    twin_rows = pd.read_csv("features/vision/verite/verite_twin_embedding_manifest.csv")
    captions = pd.read_csv("features/vision/verite/verite_caption_embedding_manifest.csv")
    s_img, s_cap, s_twin = (np.load(drive / SIGLIP[k]) for k in ("image", "caption", "twin"))
    clip_manifest = pd.read_csv(drive / CLIP_DIR / "manifest.csv")
    assert clip_manifest["image_file"].tolist() == twins["original_filename"].tolist()
    clip_orig = np.load(drive / CLIP_DIR / "originals.npy").astype(np.float32)
    clip_twin = np.load(drive / CLIP_DIR / "twins.npy").astype(np.float32)

    split_of = {os.path.basename(p): s for p, s in zip(verite["image_path"], verite["split"])}
    twin_row_of = dict(zip(twin_rows["original_embedding_index"], twin_rows["embedding_row"]))
    img = twins[["original_filename", "original_embedding_index"]].rename(columns={"original_filename": "image_file"})
    img["twin_row"] = img["original_embedding_index"].map(twin_row_of)
    img["split"] = img["image_file"].map(split_of)
    img["clip_index"] = np.arange(len(img))
    img = img.merge(auth[["image_file", "original_authenticity_score", "twin_authenticity_score"]], on="image_file")

    cap = captions[captions["label"].isin(["true", "miscaptioned"])][["embedding_index", "image_filename", "label"]]
    cap = cap.rename(columns={"image_filename": "image_file"}).merge(img, on="image_file", how="inner")
    assert (cap.groupby("image_file")["label"].apply(sorted).apply(tuple) == ("miscaptioned", "true")).all()
    cap["y"] = (cap["label"] == "true").astype(int)
    cap["sim_original"] = (s_img[cap["original_embedding_index"]] * s_cap[cap["embedding_index"]]).sum(1)
    cap["sim_twin"] = (s_twin[cap["twin_row"]] * s_cap[cap["embedding_index"]]).sum(1)
    return cap.reset_index(drop=True), clip_orig, clip_twin


def features(cap: pd.DataFrame, model: str, version: str, emb: dict | None = None) -> np.ndarray:
    sim = cap[f"sim_{version}"].to_numpy()
    if model == "decoupled":
        return sim[:, None]
    a = cap[f"{version}_authenticity_score"].to_numpy()
    base = np.c_[sim, a, sim * a]
    return base if model == "fused" else np.c_[base, emb[version]]


def fit_and_score(cap: pd.DataFrame, train: np.ndarray, held: np.ndarray, clip_orig: np.ndarray,
                  clip_twin: np.ndarray, seed: int, components: int = 16) -> pd.DataFrame:
    train_photos = cap.loc[train, "clip_index"].unique()
    pca = PCA(n_components=components, random_state=seed).fit(clip_orig[np.sort(train_photos)])
    emb = {"original": pca.transform(clip_orig[cap["clip_index"]]), "twin": pca.transform(clip_twin[cap["clip_index"]])}
    rows = []
    for model in MODELS:
        fitted = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        fitted.fit(features(cap, model, "original", emb)[train], cap.loc[train, "y"])
        for version in ("original", "twin"):
            part = cap[held].assign(prob=fitted.predict_proba(features(cap, model, version, emb)[held])[:, 1])
            wide = part.pivot(index="image_file", columns="label", values="prob")
            for f, r in wide.iterrows():
                rows.append({"model": model, "version": version, "image_file": f,
                             "p_true_caption": r["true"], "p_miscaptioned": r["miscaptioned"],
                             "picks_true_caption": bool(r["true"] > r["miscaptioned"])})
    return pd.DataFrame(rows)


def fixed_split(cap: pd.DataFrame, clip_orig: np.ndarray, clip_twin: np.ndarray, seed: int = 42) -> pd.DataFrame:
    train = (cap["split"] == "train").to_numpy()
    pairs = fit_and_score(cap, train, np.ones(len(cap), bool), clip_orig, clip_twin, seed)
    return pairs.assign(split=pairs["image_file"].map(dict(zip(cap["image_file"], cap["split"]))))


def cross_validate(cap: pd.DataFrame, clip_orig: np.ndarray, clip_twin: np.ndarray, seed: int,
                   folds: int = 5, components: int = 16) -> pd.DataFrame:
    photos = np.array(sorted(cap["image_file"].unique()))
    order = np.random.default_rng(seed).permutation(len(photos))
    fold_of = {photos[i]: k % folds for k, i in enumerate(order)}
    cap_fold = cap["image_file"].map(fold_of).to_numpy()
    parts = []
    for k in range(folds):
        held = cap_fold == k
        parts.append(fit_and_score(cap, ~held, held, clip_orig, clip_twin, seed, components).assign(fold=k))
    return pd.concat(parts, ignore_index=True).assign(seed=seed)


def changed(pairs: pd.DataFrame, model: str) -> pd.Series:
    wide = pairs[pairs["model"] == model].pivot(index="image_file", columns="version", values="picks_true_caption")
    return wide["original"] != wide["twin"]


def accuracy(pairs: pd.DataFrame, model: str, version: str) -> float:
    return float(pairs[(pairs["model"] == model) & (pairs["version"] == version)]["picks_true_caption"].mean())


def alr_table(pairs: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for model in MODELS:
        c = changed(pairs, model).to_numpy().astype(float)
        boot = c[rng.integers(0, len(c), (n_boot, len(c)))].mean(1) * 100
        rows.append({"model": model, "photos": len(c), "original_accuracy": accuracy(pairs, model, "original"),
                     "twin_accuracy": accuracy(pairs, model, "twin"), "changed_verdicts": int(c.sum()),
                     "ALR_percent": c.mean() * 100, "ci_low": np.percentile(boot, 2.5),
                     "ci_high": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


def mcnemar(pairs: pd.DataFrame, a: str = "decoupled", b: str = "fused") -> dict:
    ca, cb = changed(pairs, a), changed(pairs, b).reindex(changed(pairs, a).index)
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"model_a": a, "model_b": b, "photos": len(ca), "changed_a": int(ca.sum()), "changed_b": int(cb.sum()),
            "both": int((ca & cb).sum()), "only_a": only_a, "only_b": only_b, "p_exact": float(p)}


def shift_analysis(pairs: pd.DataFrame, auth: pd.DataFrame) -> pd.DataFrame:
    shift = (auth.set_index("image_file")["twin_authenticity_score"]
             - auth.set_index("image_file")["original_authenticity_score"])
    orig = pairs[pairs["version"] == "original"].assign(
        gap=lambda d: (d["p_true_caption"] - d["p_miscaptioned"]).abs())
    rows = [{"group": "all photos", "model": "-", "photos": len(shift), "median_shift": shift.median(),
             "share_abs_shift_over_0.1": (shift.abs() > 0.1).mean(), "median_caption_gap": np.nan}]
    dec = changed(pairs, "decoupled")
    for model in MODELS:
        c = changed(pairs, model)
        gaps = orig[orig["model"] == model].set_index("image_file")["gap"]
        groups = {f"changed by {model}": c[c].index}
        if model != "decoupled":
            groups[f"changed by {model} only (not decoupled)"] = c[c & ~dec.reindex(c.index)].index
        groups[f"all photos ({model} caption gap)"] = c.index
        for g, files in groups.items():
            s = shift.reindex(files)
            rows.append({"group": g, "model": model, "photos": len(files),
                         "median_shift": s.median() if len(s) else np.nan,
                         "share_abs_shift_over_0.1": (s.abs() > 0.1).mean() if len(s) else np.nan,
                         "median_caption_gap": gaps.reindex(files).median() if len(files) else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    ap.add_argument("--out", default="results/F10")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    auth = pd.read_csv("results/F3/F3_verite_authenticity_scores.csv")
    cap, clip_orig, clip_twin = build_captions(args.drive, auth)

    f3 = pd.read_csv("results/F3/F3_pair_verdicts.csv")
    mine = fixed_split(cap, clip_orig, clip_twin)
    both = f3.merge(mine, on=["model", "version", "image_file", "split"], suffixes=("_f3", "_f10"))
    test = mine[mine["split"] == "test"]
    check = pd.DataFrame([{
        "model": m, "rows": len(d),
        "max_abs_diff_p": float(np.abs(np.r_[d["p_true_caption_f3"] - d["p_true_caption_f10"],
                                             d["p_miscaptioned_f3"] - d["p_miscaptioned_f10"]]).max()),
        "same_verdict": float((d["picks_true_caption_f3"] == d["picks_true_caption_f10"]).mean()),
        "test_ALR_percent_f10": changed(test, m).mean() * 100}
        for m, d in ((m, both[both["model"] == m]) for m in MODELS)])
    check.to_csv(out / "leakage_f3_check.csv", index=False)
    print(check.round(4).to_string(index=False), flush=True)

    tables, tests, shifts, pairs_all = [], [], [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        pairs = cross_validate(cap, clip_orig, clip_twin, seed, args.folds)
        pairs_all.append(pairs)
        tables.append(alr_table(pairs, args.boot).assign(seed=seed))
        tests += [{**mcnemar(pairs, "decoupled", b), "seed": seed} for b in ("fused", "fused + image embedding")]
        shifts.append(shift_analysis(pairs, auth).assign(seed=seed))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "leakage_cv.csv", index=False)
    table.groupby("model", sort=False).agg(
        seeds=("seed", "size"), ALR_mean=("ALR_percent", "mean"), ALR_sd=("ALR_percent", "std"),
        changed_mean=("changed_verdicts", "mean"), original_accuracy_mean=("original_accuracy", "mean"),
        twin_accuracy_mean=("twin_accuracy", "mean")).reset_index().to_csv(out / "leakage_cv_seeds.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "leakage_cv_mcnemar.csv", index=False)
    pd.concat(shifts, ignore_index=True).to_csv(out / "leakage_cv_shift.csv", index=False)
    pd.concat(pairs_all, ignore_index=True).to_csv(out / "leakage_cv_pairs.csv", index=False)
    print(table.round(3).to_string(index=False))
    print(pd.DataFrame(tests).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
