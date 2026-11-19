"""Similarity baseline (MUSE-style): verdict from claim-evidence similarity features + LightGBM.

What it does
    Turns each (claim, evidence snippets) record into 12 similarity features and
    trains a 3-class LightGBM on them. It sees *how similar* the evidence is to the
    claim, never what the evidence says, which is the "similarity over factuality"
    shortcut (Papadopoulos et al., WACV 2025). Lexical TF-IDF similarity is used, so
    it runs on a CPU in minutes and on Arabic text too (no embedding model).

Features (per record)
    n_snippets, claim_tokens; TF-IDF cosine claim-vs-snippet: max, mean, mean of top 3,
    min; token Jaccard: max, mean; share of claim tokens found in any snippet; share of
    claim numbers found in any snippet; mean snippet length; has_evidence.

Training
    LightGBM multiclass, class-balanced, 0.05 learning rate, early stopping on the
    validation file (50 rounds), bagging 0.8 / feature fraction 0.8 so the seed
    matters; seeds 42, 43, 44 -> one predictions file per seed.

Inputs
    --train / --val: labelled shared-format files; --predict: any shared-format files.

Outputs
    ``<out_dir>/preds_<name>_s<seed>.jsonl`` for each seed (CLAUDE.md §5 format);
    ``--name`` defaults to ``similarity`` (e.g. ``avt_similarity`` for AVeriTeC).

How to run
    python -m src.eval.baselines.similarity \
        --train data/processed/en/multifc_train.jsonl --val data/processed/en/multifc_val.jsonl \
        --predict data/processed/en/multifc_test.jsonl data/copies/en/test.jsonl --out-dir results/F4
    Add --limit 200 for a quick check.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from src.eval.baselines.common import (evidence_texts, label_ids, load_records, to_preds,
                                       tokens, write_preds)
from src.vision.tracking import finish, log_summary, start_run

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
FEATURE_NAMES = ("n_snippets", "claim_tokens", "cos_max", "cos_mean", "cos_top3", "cos_min",
                 "jac_max", "jac_mean", "claim_coverage", "number_coverage",
                 "snippet_len_mean", "has_evidence")
SEEDS = (42, 43, 44)


def fit_vectorizer(records: list[dict]) -> TfidfVectorizer:
    """TF-IDF over claims + snippets of the training records only."""
    texts = [r["claim"] for r in records] + [t for r in records for t in evidence_texts(r)]
    return TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", sublinear_tf=True, min_df=2).fit(texts)


def record_features(rec: dict, vec: TfidfVectorizer) -> list[float]:
    """The 12 similarity features of one record (see module docstring)."""
    snippets = evidence_texts(rec)
    claim_tok = set(tokens(rec["claim"]))
    if not snippets:
        return [0, len(claim_tok), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    m = vec.transform([rec["claim"]] + snippets)
    cos = (m[1:] @ m[0].T).toarray().ravel()  # rows are L2-normalised, so dot = cosine
    snip_tok = [set(tokens(s)) for s in snippets]
    jac = [len(claim_tok & s) / max(1, len(claim_tok | s)) for s in snip_tok]
    all_tok = set().union(*snip_tok)
    claim_nums = set(NUMBER.findall(rec["claim"]))
    ev_nums = set(NUMBER.findall(" ".join(snippets)))
    return [len(snippets), len(claim_tok), cos.max(), cos.mean(), np.sort(cos)[-3:].mean(), cos.min(),
            max(jac), float(np.mean(jac)), len(claim_tok & all_tok) / max(1, len(claim_tok)),
            len(claim_nums & ev_nums) / len(claim_nums) if claim_nums else 1.0,
            float(np.mean([len(s) for s in snip_tok])), 1]


def feature_matrix(records: list[dict], vec: TfidfVectorizer) -> np.ndarray:
    """Stack the features of many records."""
    return np.array([record_features(r, vec) for r in records], dtype=np.float32)


def train_lgbm(X_tr, y_tr, X_val, y_val, seed: int) -> lgb.LGBMClassifier:
    """Class-balanced multiclass LightGBM with early stopping on the validation set."""
    model = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=1000,
                               learning_rate=0.05, num_leaves=31, class_weight="balanced",
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               random_state=seed, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def main() -> None:
    """Build features, train one model per seed, write one predictions file per seed."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--predict", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="similarity", help="file name: preds_<name>_s<seed>.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    train, val = load_records(args.train), load_records(args.val)
    if args.limit:
        train, val = train[: args.limit], val[: args.limit]
    vec = fit_vectorizer(train)
    X_tr, X_val = feature_matrix(train, vec), feature_matrix(val, vec)
    y_tr, y_val = label_ids(train), label_ids(val)
    records = load_records(args.predict)
    X_pred = feature_matrix(records, vec)

    for seed in SEEDS:
        model = train_lgbm(X_tr, y_tr, X_val, y_val, seed)
        proba = model.predict_proba(X_pred)  # classes 0..2 = LABELS order
        out = Path(args.out_dir) / f"preds_{args.name}_s{seed}.jsonl"
        write_preds(to_preds(records, proba), out)
        val_f1 = f1_score(y_val, model.predict(X_val), average="macro")
        run = start_run(f"F4-{args.name}-s{seed}", {"seed": seed, "train": args.train,
                        "features": list(FEATURE_NAMES)}, not args.no_wandb)
        log_summary(run, {"val_macro_f1": val_f1, "best_iteration": model.best_iteration_})
        finish(run)
        print(f"similarity s{seed}: val macro-F1 {val_f1:.3f}, "
              f"{model.best_iteration_} trees -> {out}")


if __name__ == "__main__":
    main()
