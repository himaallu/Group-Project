"""Claim-only baseline: predicts the verdict from the claim text alone (no evidence).

What it does
    TF-IDF over word 1-2-grams of the claim + class-balanced multinomial logistic
    regression. C is chosen on the validation file (macro-F1) from 0.1 / 1 / 10, then
    the model is refit on the training file. It never reads the evidence, so it shows
    how far "how plausible the claim sounds" gets (the shortcut twins are built to
    expose): its prediction for a ``_B``/``_C`` twin equals the original's, because
    those twins keep the claim and change only the evidence.

Inputs
    --train / --val: shared-format files with labels (any language; for Arabic pass the
    X-FACT AR files). --predict: one or more shared-format files (originals and/or twins).

Outputs
    --out: predictions file ``results/<task>/preds_claim_only.jsonl`` (CLAUDE.md §5).
    Logistic regression is deterministic, so one run (seed 42) is enough.

How to run
    python -m src.eval.baselines.claim_only \
        --train data/processed/en/multifc_train.jsonl --val data/processed/en/multifc_val.jsonl \
        --predict data/processed/en/multifc_test.jsonl data/copies/en/test.jsonl \
        --out results/F4/preds_claim_only.jsonl
    Add --limit 10 for a quick check (10 training records per label).
"""
from __future__ import annotations

import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from src.eval.baselines.common import full_proba, label_ids, load_records, to_preds, write_preds
from src.vision.tracking import finish, log_summary, start_run

C_GRID = (0.1, 1.0, 10.0)
SEED = 42


def build(C: float):
    """TF-IDF (word 1-2-grams) + balanced multinomial logistic regression."""
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000),
        LogisticRegression(C=C, max_iter=3000, class_weight="balanced"),
    )


def claims(records: list[dict]) -> list[str]:
    """The claim texts."""
    return [r["claim"] for r in records]


def fit_claim_only(train: list[dict], val: list[dict]):
    """Pick C on ``val`` by macro-F1, refit on ``train``; returns (model, C, val macro-F1)."""
    y_tr, y_val = label_ids(train), label_ids(val)
    scores = {}
    for C in C_GRID:
        pred = build(C).fit(claims(train), y_tr).predict(claims(val))
        scores[C] = f1_score(y_val, pred, average="macro")
    best = max(scores, key=scores.get)
    return build(best).fit(claims(train), y_tr), best, scores[best]


def predict(model, records: list[dict]) -> list[dict]:
    """Predictions for any shared-format records."""
    return to_preds(records, full_proba(model, claims(records)))


def main() -> None:
    """Fit on --train/--val and write predictions for every --predict file."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--predict", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    train, val = load_records(args.train), load_records(args.val)
    if args.limit:  # keep N per label so every class is present
        train = [r for lab in ("supported", "contradicted", "unverified")
                 for r in [x for x in train if x["label"] == lab][: args.limit]]
    model, C, val_f1 = fit_claim_only(train, val)
    records = load_records(args.predict)
    write_preds(predict(model, records), args.out)

    run = start_run(f"F4-claim_only-s{SEED}", {"seed": SEED, "C": C, "train": args.train,
                    "features": "tfidf word 1-2gram"}, not args.no_wandb)
    log_summary(run, {"val_macro_f1": val_f1})
    finish(run)
    print(f"claim-only: C={C}, val macro-F1 {val_f1:.3f}, {len(records)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
