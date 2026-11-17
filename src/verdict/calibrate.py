"""Calibrate the verdict head: temperature scaling + an "Unverified" threshold (H4).

What it does
    1. Temperature scaling: divides the verdict head's logits by one number T,
       chosen to minimise negative log-likelihood (NLL) on the validation split.
       T > 1 softens over-confident probabilities; the argmax never changes.
    2. "Unverified" threshold: if the highest calibrated probability is below tau,
       the verdict becomes ``unverified``. tau is chosen to maximise validation
       macro-F1 (tau = 0 means "no threshold").
    Both values are fitted on validation data only and then applied unchanged to
    every test set (MultiFC, AVeriTeC, X-FACT AR).

Inputs
    A logits file (JSON Lines), one line per validation original:
        {"id": "multifc_000123", "gold": "supported", "logits": [2.1, -0.3, 0.4]}
    Logits are in the order ``supported, contradicted, unverified`` (= LABELS).

Outputs
    ``models/verdict_calib.json``: {"T": 1.83, "tau": 0.45, "val_nll_before": ...,
    "val_nll_after": ..., "val_macro_f1": ..., "n_val": ...}

How to run
    python -m src.verdict.calibrate --val-logits results/H3/val_logits.jsonl \
        --out models/verdict_calib.json

    In code:
        from src.verdict.calibrate import load_settings, apply_calibration
        s = load_settings("models/verdict_calib.json")
        preds, probs = apply_calibration(logits, s["T"], s["tau"])
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.eval.metrics import LABELS, ece, macro_f1

UNVERIFIED = "unverified"


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Row-wise softmax of ``logits / T``.

    Args:
        logits: (n, 3) array of raw scores.
        T: temperature (> 0).

    Returns:
        (n, 3) array of probabilities; each row sums to 1.
    """
    z = np.asarray(logits, dtype=float) / T
    z = z - z.max(axis=1, keepdims=True)  # subtract the row max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _gold_index(gold: Sequence[str]) -> np.ndarray:
    """Turn label strings into column indices in LABELS order."""
    return np.array([LABELS.index(g) for g in gold])


def nll(logits: np.ndarray, gold: Sequence[str], T: float = 1.0) -> float:
    """Mean negative log-likelihood of the gold labels at temperature T.

    Args:
        logits: (n, 3) raw scores.
        gold: n gold labels.
        T: temperature.

    Returns:
        Mean NLL (lower is better).
    """
    z = np.asarray(logits, dtype=float) / T
    z = z - z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(z)), _gold_index(gold)].mean())


def fit_temperature(logits: np.ndarray, gold: Sequence[str],
                    t_min: float = 0.05, t_max: float = 10.0) -> float:
    """Find the temperature that minimises validation NLL.

    A coarse log-spaced grid search followed by a fine grid around the best point.
    NLL is smooth and has one minimum in T, so this is reliable and needs no scipy.

    Args:
        logits: (n, 3) validation logits.
        gold: n validation gold labels.
        t_min, t_max: search range for T.

    Returns:
        The fitted temperature T.
    """
    coarse = np.geomspace(t_min, t_max, 100)
    best = coarse[int(np.argmin([nll(logits, gold, t) for t in coarse]))]
    # Fine search between the neighbouring coarse points (one coarse step ~5.5%).
    fine = np.linspace(max(t_min, best / 1.06), min(t_max, best * 1.06), 101)
    return float(fine[int(np.argmin([nll(logits, gold, t) for t in fine]))])


def apply_threshold(probs: np.ndarray, tau: float) -> list[str]:
    """Predicted labels: argmax, but ``unverified`` when max probability < tau.

    Args:
        probs: (n, 3) probabilities in LABELS order.
        tau: threshold in [0, 1); 0 means argmax only.

    Returns:
        List of n label strings.
    """
    probs = np.asarray(probs, dtype=float)
    preds = [LABELS[i] for i in probs.argmax(axis=1)]
    low = probs.max(axis=1) < tau
    return [UNVERIFIED if is_low else p for p, is_low in zip(preds, low)]


def fit_unverified_threshold(probs: np.ndarray, gold: Sequence[str],
                             grid: Sequence[float] | None = None) -> float:
    """Pick the "Unverified" threshold that maximises validation macro-F1.

    Args:
        probs: (n, 3) *calibrated* validation probabilities.
        gold: n gold labels.
        grid: candidate thresholds; default 0 (no threshold) plus 0.30, 0.31, ..., 0.90.

    Returns:
        Best tau. On ties the smallest tau wins (change as few verdicts as possible).
    """
    if grid is None:
        grid = [0.0] + [round(x, 2) for x in np.arange(0.30, 0.901, 0.01)]
    scores = [macro_f1(list(gold), apply_threshold(probs, t)) or 0.0 for t in grid]
    return float(grid[int(np.argmax(scores))])  # argmax returns the first (smallest) best


def apply_calibration(logits: np.ndarray, T: float, tau: float) -> tuple[list[str], np.ndarray]:
    """Apply fitted settings to new logits.

    Args:
        logits: (n, 3) raw scores.
        T: fitted temperature.
        tau: fitted Unverified threshold.

    Returns:
        (predicted labels, calibrated probabilities). The probabilities are not
        changed by the threshold; pass the labels as ``pred`` to ``metrics.ece``.
    """
    probs = softmax(logits, T)
    return apply_threshold(probs, tau), probs


def fit(logits: np.ndarray, gold: Sequence[str]) -> dict:
    """Fit T then tau on validation data and return both with summary numbers.

    Args:
        logits: (n, 3) validation logits.
        gold: n validation gold labels.

    Returns:
        Dict with T, tau, NLL and ECE before/after, validation macro-F1 after.
    """
    logits = np.asarray(logits, dtype=float)
    T = fit_temperature(logits, gold)
    probs = softmax(logits, T)
    tau = fit_unverified_threshold(probs, gold)
    preds = apply_threshold(probs, tau)
    return {
        "T": T,
        "tau": tau,
        "n_val": len(gold),
        "val_nll_before": nll(logits, gold, 1.0),
        "val_nll_after": nll(logits, gold, T),
        "val_ece_before": ece(gold, softmax(logits)),
        "val_ece_after": ece(gold, probs, pred=preds),
        "val_macro_f1_before": macro_f1(list(gold), apply_threshold(softmax(logits), 0.0)),
        "val_macro_f1_after": macro_f1(list(gold), preds),
    }


def load_logits(path: str | Path) -> tuple[list[str], np.ndarray, list[str]]:
    """Read a logits JSONL file (format in the module docstring).

    Returns:
        (ids, (n, 3) logits array, gold labels).
    """
    ids, logits, gold = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                ids.append(rec["id"])
                logits.append(rec["logits"])
                gold.append(rec["gold"])
    return ids, np.array(logits, dtype=float), gold


def save_settings(settings: dict, path: str | Path) -> None:
    """Write calibration settings as pretty JSON (creates the folder)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def load_settings(path: str | Path) -> dict:
    """Read calibration settings written by ``save_settings``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    """CLI: fit on a validation logits file and save the settings."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--val-logits", required=True, help="validation logits JSONL")
    ap.add_argument("--out", default="models/verdict_calib.json")
    args = ap.parse_args()
    _, logits, gold = load_logits(args.val_logits)
    settings = fit(logits, gold)
    save_settings(settings, args.out)
    print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
