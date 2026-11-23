from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ARCHS = ("resnet50", "vit_b16")
SEEDS = (42, 43, 44)
METRICS = ("accuracy", "precision", "recall", "f1", "auroc")


def seed_tables(f5: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for arch in ARCHS:
        for seed in SEEDS:
            path = f5 / f"train_{arch}_s{seed}.json"
            if not path.exists():
                print("missing", path)
                continue
            run = json.loads(path.read_text())
            for split in ("validation", "test"):
                rows.append({"arch": arch, "seed": seed, "split": split, **{m: run[split][m] for m in METRICS}})
    per_run = pd.DataFrame(rows)
    summary = per_run.groupby(["arch", "split"], sort=False).agg(
        seeds=("seed", "size"), **{f"{m}_{s}": (m, f) for m in METRICS for s, f in (("mean", "mean"), ("sd", "std"))})
    return per_run, summary.reset_index()


def test_scores(seed: int, f6: Path, preds: Path) -> pd.DataFrame | None:
    if seed == 42:
        p = pd.read_csv(f6 / "predictions.csv")
        return p[p["split"] == "test"].rename(columns={f"p_ai_{a}": a for a in ARCHS})
    parts = []
    for arch in ARCHS:
        path = preds / f"preds_{arch}_s{seed}.csv"
        if not path.exists():
            print("missing", path)
            return None
        p = pd.read_csv(path)
        parts.append(p[p["split"] == "test"].set_index("md5")[["class_name", "p_ai"]].rename(columns={"p_ai": arch}))
    return parts[0].join(parts[1][[ARCHS[1]]]).reset_index()


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    au_a, au_b, diff = [], [], []
    while len(diff) < n_boot:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        x, z = roc_auc_score(y[i], a[i]), roc_auc_score(y[i], b[i])
        au_a.append(x), au_b.append(z), diff.append(z - x)
    diff = np.array(diff)
    p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
    return {"auroc_resnet50": roc_auc_score(y, a), "resnet50_low": np.percentile(au_a, 2.5),
            "resnet50_high": np.percentile(au_a, 97.5), "auroc_vit_b16": roc_auc_score(y, b),
            "vit_b16_low": np.percentile(au_b, 2.5), "vit_b16_high": np.percentile(au_b, 97.5),
            "difference": roc_auc_score(y, b) - roc_auc_score(y, a), "difference_low": np.percentile(diff, 2.5),
            "difference_high": np.percentile(diff, 97.5), "p_bootstrap": p, "resamples": n_boot}


def cnn_vs_vit(f6: Path, preds: Path, n_boot: int) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        t = test_scores(seed, f6, preds)
        if t is None:
            continue
        y = (t["class_name"] == "fake").to_numpy().astype(int)
        rows.append({"seed": seed, "test_images": len(t),
                     **paired_bootstrap(y, t["resnet50"].to_numpy(), t["vit_b16"].to_numpy(), n_boot)})
    return pd.DataFrame(rows)


def choose_cutoff(p_val: np.ndarray, y_val: np.ndarray) -> float:
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    acc = np.array([accuracy_score(y_val, p_val >= t) for t in grid])
    best = np.flatnonzero(acc == acc.max())
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])


def at_cutoff(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = p >= t
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "real_flagged": float(pred[y == 0].mean())}


def cutoff_table(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, p in runs.items():
        val, test = p[p["split"] == "validation"], p[p["split"] == "test"]
        yv, yt = (val["class_name"] == "fake").to_numpy(), (test["class_name"] == "fake").to_numpy()
        t = choose_cutoff(val["p_ai"].to_numpy(), yv)
        for cut, label in ((0.5, "0.5"), (t, "chosen on validation")):
            rows.append({"run": name, "cutoff_rule": label, "cutoff": cut,
                         **{f"val_{k}": v for k, v in at_cutoff(yv, val["p_ai"].to_numpy(), cut).items()},
                         "val_auroc": roc_auc_score(yv, val["p_ai"]),
                         **{f"test_{k}": v for k, v in at_cutoff(yt, test["p_ai"].to_numpy(), cut).items()},
                         "test_auroc": roc_auc_score(yt, test["p_ai"])})
    return pd.DataFrame(rows)


def resnet_runs(f6: Path, preds: Path) -> dict[str, pd.DataFrame]:
    p = pd.read_csv(f6 / "predictions.csv")
    runs = {"resnet50_s42 (F5)": p.rename(columns={"p_ai_resnet50": "p_ai"})[["md5", "split", "class_name", "p_ai"]]}
    for tag, label in (("resnet50_s42_imagenetbn", "resnet50_s42 + ImageNet BatchNorm statistics (no retraining)"),
                       ("resnet50_s42_freezebn", "resnet50_s42 retrained with frozen BatchNorm")):
        path = preds / f"preds_{tag}.csv"
        if path.exists():
            runs[label] = pd.read_csv(path)
        else:
            print("missing", path)
    return runs


def bn_check(f5: Path, out: Path) -> pd.DataFrame:
    files = {"F5 (pushed)": f5 / "train_resnet50_s42.json", "F5 checkpoint re-scored": out / "eval_resnet50_s42_check.json",
             "F5 checkpoint + ImageNet BatchNorm statistics": out / "eval_resnet50_s42_imagenetbn.json",
             "retrained with frozen BatchNorm": out / "train_resnet50_s42_freezebn.json"}
    rows = []
    for name, path in files.items():
        if not path.exists():
            print("missing", path)
            continue
        run = json.loads(path.read_text())
        rows.append({"run": name, **{f"{s}_{m}": run[s][m] for s in ("validation", "test") for m in METRICS},
                     "val_acc_BigGAN": run["validation"].get("acc_BigGAN")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f5", default="results/F5")
    ap.add_argument("--f6", default="results/F6")
    ap.add_argument("--out", default="results/F10")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    f5, f6, out = Path(args.f5), Path(args.f6), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_run, summary = seed_tables(f5)
    per_run.to_csv(out / "seeds_per_run.csv", index=False)
    summary.to_csv(out / "seeds_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    versus = cnn_vs_vit(f6, out, args.boot)
    versus.to_csv(out / "cnn_vs_vit_seeds.csv", index=False)
    print(versus.round(3).to_string(index=False))
    cut = cutoff_table(resnet_runs(f6, out))
    cut.to_csv(out / "cutoff_validation.csv", index=False)
    print(cut.round(3).to_string(index=False))
    bn = bn_check(f5, out)
    bn.to_csv(out / "bn_check.csv", index=False)
    print(bn.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
