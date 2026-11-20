"""Why an MLP verdict head? Compare classifiers on the same 10 features (H7, model selection).

What it does
    Builds the verdict-head features (``src/verdict/features.py``, zero-shot NLI +
    reranker relevance + entity match) once, then trains alternative classifiers on
    MultiFC train and scores MultiFC test originals + test twins with the shared
    metrics (``src/eval/metrics.py``):
        lr            multinomial logistic regression (class-balanced)
        lr_aug        same, train twins added as extra labelled examples
        lgbm          LightGBM (class-balanced), seeds 42/43/44
        lgbm_aug      same, train twins added as extra labelled examples
    Twins as extra examples is the closest a non-differentiable model gets to the twin
    loss: _B -> contradicted, _C -> unverified, _P keeps the label; it cannot express
    "keep the same answer as the original" (the KL term). The MLP rows (``notwin`` and
    ``twin``, H5) are read from ``results/H5/summary.csv`` and use the same features,
    splits and test sets, so all rows are comparable.

Inputs
    data/processed/en/multifc_{train,val,test}.jsonl, data/copies/en/{train,test}.jsonl,
    features/nli_zeroshot/{multifc_*,copies_en/*}.parquet,
    features/relevance_{en,copies_en}.parquet, features/entities_en.jsonl,
    results/H5/summary.csv

Outputs
    results/H7/preds_<model>_s<seed>.jsonl, results/H7/verdict_head_selection.csv
    (mean ± sd over seeds, plus training time and number of parameters)

How to run (CPU, ~2 min)
    python -m src.eval.model_selection
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.loaders import join_gold, load_jsonl
from src.eval.metrics import LABELS, compute_all
from src.verdict.features import FEATURES, build_features, load_entities
from src.verdict.train import write_jsonl

EN, CP, NLI = "data/processed/en", "data/copies/en", "features/nli_zeroshot"
OUT = Path("results/H7")
COLS = ("macro_f1", "auroc", "ece", "esr", "unrelated_rate", "paraphrase_consistency")


def load_tables() -> dict[str, pd.DataFrame]:
    """Feature tables for train originals, train twins, test originals + twins."""
    nli = pd.concat([pd.read_parquet(f"{NLI}/multifc_{s}.parquet") for s in ("train", "test")]
                    + [pd.read_parquet(f"{NLI}/copies_en/{s}.parquet") for s in ("train", "test")],
                    ignore_index=True)
    rel = pd.concat([pd.read_parquet("features/relevance_en.parquet"),
                     pd.read_parquet("features/relevance_copies_en.parquet")], ignore_index=True)
    ent = load_entities("features/entities_en.jsonl")
    twins = load_jsonl(f"{CP}/train.jsonl")
    test = load_jsonl(f"{EN}/multifc_test.jsonl") + load_jsonl(f"{CP}/test.jsonl")
    return {"train": build_features(load_jsonl(f"{EN}/multifc_train.jsonl"), nli, rel, ent),
            "twins": build_features(twins, nli, rel, ent),
            "test": build_features(test, nli, rel, ent), "test_records": test}


def make_model(kind: str, seed: int):
    """An sklearn-style classifier for one row of the comparison."""
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=2000))
    from lightgbm import LGBMClassifier
    return LGBMClassifier(class_weight="balanced", n_estimators=300, learning_rate=0.05,
                          num_leaves=31, random_state=seed, verbose=-1)


def n_params(model, kind: str) -> int:
    """Rough number of learned parameters (LR weights; LightGBM tree leaves)."""
    if kind == "lr":
        lr = model[-1]
        return int(lr.coef_.size + lr.intercept_.size)
    return int(sum(t["num_leaves"] for t in model.booster_.dump_model()["tree_info"]))


def run() -> pd.DataFrame:
    """Train every variant, write predictions, return the comparison table."""
    t = load_tables()
    X_test = t["test"][list(FEATURES)].to_numpy()
    rows = []
    for name, kind, aug, seeds in (("lr", "lr", False, [42]), ("lr_aug", "lr", True, [42]),
                                   ("lgbm", "lgbm", False, [42, 43, 44]),
                                   ("lgbm_aug", "lgbm", True, [42, 43, 44])):
        train = pd.concat([t["train"], t["twins"]]) if aug else t["train"]
        X, y = train[list(FEATURES)].to_numpy(), train["label"].map(LABELS.index).to_numpy()
        for seed in seeds:
            model = make_model(kind, seed)
            start = time.time()
            model.fit(X, y)
            secs = time.time() - start
            probs = np.zeros((len(X_test), 3))
            probs[:, model.classes_] = model.predict_proba(X_test)
            preds = [{"id": i, "pred_label": LABELS[int(p.argmax())],
                      "probs": {lab: float(v) for lab, v in zip(LABELS, p / p.sum())}}
                     for i, p in zip(t["test"]["id"], probs)]
            write_jsonl(preds, OUT / f"preds_{name}_s{seed}.jsonl")
            df, _ = join_gold(preds, t["test_records"])
            rows.append({"model": name, "seed": seed, "train_seconds": round(secs, 2),
                         "n_params": n_params(model, kind), **compute_all(df)})
            print(f"{name} s{seed}: macro-F1 {rows[-1]['macro_f1']:.3f} ESR {rows[-1]['esr']:.3f}",
                  flush=True)
    res = pd.DataFrame(rows)
    table = res.groupby("model")[list(COLS)].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table["train_seconds"] = res.groupby("model")["train_seconds"].mean()
    table["n_params"] = res.groupby("model")["n_params"].mean()
    # MLP rows from H5 (same features, splits, test sets): 10*64+64 + 64*3+3 = 899 parameters
    h5 = pd.read_csv("results/H5/summary.csv", index_col=0).loc[["notwin", "twin"]]
    h5 = h5.rename(index={"notwin": "mlp", "twin": "mlp_twinloss"})
    h5["n_params"] = len(FEATURES) * 64 + 64 + 64 * 3 + 3
    table = pd.concat([table, h5[[c for c in table.columns if c in h5.columns]]])
    return table.round(4)


def main() -> None:
    """CLI: run and save the table."""
    OUT.mkdir(parents=True, exist_ok=True)
    table = run()
    table.to_csv(OUT / "verdict_head_selection.csv")
    pd.set_option("display.width", 250)
    print(table.to_string())


if __name__ == "__main__":
    main()
