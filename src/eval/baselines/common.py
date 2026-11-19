"""Shared helpers for the baselines in ``src/eval/baselines/`` (owner: Feba).

Every baseline reads shared-format JSON Lines files (CLAUDE.md §4: originals or
twins, any language) and writes predictions in the CLAUDE.md §5 format:
``{"id", "pred_label", "probs": {"supported", "contradicted", "unverified"}}``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from src.eval.loaders import LABELS, load_jsonl

TOKEN = re.compile(r"\w+", re.UNICODE)  # works for English and Arabic script


def tokens(text: str) -> list[str]:
    """Lower-cased word tokens (Unicode letters and digits)."""
    return TOKEN.findall(text.lower())


def load_records(paths: list[str | Path]) -> list[dict]:
    """Concatenate the records of several shared-format files."""
    records = []
    for path in paths:
        records += load_jsonl(path)
    return records


def label_ids(records: list[dict]) -> np.ndarray:
    """Gold labels as integers in the order of ``LABELS``."""
    return np.array([LABELS.index(r["label"]) for r in records])


def evidence_texts(record: dict) -> list[str]:
    """The snippet texts of one record (empty list if it has no evidence)."""
    return [snip["text"] for snip in record.get("evidence") or []]


def to_preds(records: list[dict], proba: np.ndarray) -> list[dict]:
    """Turn an (n, 3) probability matrix (columns in ``LABELS`` order) into predictions."""
    preds = []
    for rec, p in zip(records, proba):
        p = np.asarray(p, dtype=float) / float(np.sum(p))
        preds.append({"id": rec["id"], "pred_label": LABELS[int(np.argmax(p))],
                      "probs": {lab: float(v) for lab, v in zip(LABELS, p)}})
    return preds


def write_preds(preds: list[dict], path: str | Path) -> None:
    """Write predictions as JSON Lines (creates the folder)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pred in preds:
            f.write(json.dumps(pred) + "\n")


def full_proba(model, X) -> np.ndarray:
    """``predict_proba`` with one column per label in ``LABELS`` order.

    A label missing from the training data (e.g. no ``unverified`` claims) gets 0.
    """
    proba = model.predict_proba(X)
    out = np.zeros((proba.shape[0], len(LABELS)))
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = proba[:, j]
    return out
