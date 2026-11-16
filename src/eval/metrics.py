"""All TwinCheck evaluation metrics (CLAUDE.md §5). Everyone calls these functions.

What it does
    Plain functions that turn predictions into numbers. No model is loaded here:
    metrics that need a model (faithfulness, claim matching) take that model's
    outputs, or a similarity function, as input.

Inputs
    Most twin metrics take the joined table from ``src.eval.loaders.join_gold`` with
    columns ``id, base_id, twin_type, language, gold, pred, p_supported,
    p_contradicted, p_unverified``. ``twin_type`` is None for original claims and one
    of ``P A M B C V`` for twins.

Outputs
    Floats in [0, 1] (or None when there is nothing to measure), or small dicts.

How to run
    from src.eval.loaders import load_preds, load_jsonl, join_gold
    from src.eval.metrics import compute_all
    df, _ = join_gold(load_preds("results/H5/preds_ours.jsonl"),
                      load_jsonl("data/processed/en/multifc_test.jsonl")
                      + load_jsonl("data/copies/en/test.jsonl"))
    print(compute_all(df))

Definitions (all rates are fractions; multiply by 100 for tables)
    macro-F1 / AUROC / ECE  on original claims only.
    ESR                     share of ``_B`` twins predicted ``contradicted``.
    Unrelated rate          share of ``_C`` twins predicted ``unverified``.
    Paraphrase consistency  share of ``_P`` twins whose prediction equals the
                            *prediction* for their original.
    Cross-lingual cons.     same for ``_A`` and ``_M`` twins.
    ALR                     share of ``_V`` twins whose prediction differs from
                            their original's prediction.
    Twins whose original is not in the table are skipped by the consistency metrics.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

LABELS = ("supported", "contradicted", "unverified")
PROB_COLS = tuple(f"p_{lab}" for lab in LABELS)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def originals(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the original (non-twin) rows of a joined table."""
    return df[df["twin_type"].isna()]


def twins(df: pd.DataFrame, kinds: Iterable[str]) -> pd.DataFrame:
    """Return only the twin rows whose ``twin_type`` is in ``kinds``."""
    return df[df["twin_type"].isin(list(kinds))]


def _share(mask: pd.Series) -> float | None:
    """Mean of a boolean Series, or None if it is empty."""
    return float(mask.mean()) if len(mask) else None


def _agreement_with_original(df: pd.DataFrame, kinds: Iterable[str]) -> float | None:
    """Share of twins (of the given kinds) whose prediction equals their original's.

    Args:
        df: joined table containing both originals and twins.
        kinds: twin types to include, e.g. ``("P",)`` or ``("A", "M")``.

    Returns:
        Fraction in [0, 1], or None if no twin of these kinds has its original in df.
    """
    # drop_duplicates: a bootstrap resample can contain the same original twice.
    orig_pred = originals(df).drop_duplicates("id").set_index("base_id")["pred"]
    tw = twins(df, kinds)
    tw = tw[tw["base_id"].isin(orig_pred.index)]
    if tw.empty:
        return None
    same = tw["pred"].to_numpy() == orig_pred.loc[tw["base_id"]].to_numpy()
    return float(same.mean())


def _probs_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n, 3) array of probabilities in LABELS order."""
    return df[list(PROB_COLS)].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Metrics on original claims
# --------------------------------------------------------------------------

def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float | None:
    """Macro-averaged F1 over the three labels.

    Args:
        gold: true labels.
        pred: predicted labels.

    Returns:
        Macro-F1 in [0, 1] averaged over the labels that occur in gold or pred, or
        None if the inputs are empty.
    """
    if len(gold) == 0:
        return None
    labels = [lab for lab in LABELS if lab in set(gold) | set(pred)]
    return float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))


def auroc(gold: Sequence[str], probs: np.ndarray) -> float | None:
    """Macro one-vs-rest AUROC over the three labels.

    Args:
        gold: true labels, length n.
        probs: (n, 3) probabilities in the order ``LABELS``.

    Returns:
        Mean AUROC over labels that have both positive and negative examples in gold
        (others are skipped), or None if no label qualifies.
    """
    gold = np.asarray(gold)
    probs = np.asarray(probs, dtype=float)
    scores = []
    for j, lab in enumerate(LABELS):
        y = gold == lab
        if 0 < y.sum() < len(y):  # AUROC needs both classes present
            scores.append(roc_auc_score(y, probs[:, j]))
    return float(np.mean(scores)) if scores else None


def ece(gold: Sequence[str], probs: np.ndarray, pred: Sequence[str] | None = None,
        n_bins: int = 15) -> float | None:
    """Expected calibration error with equal-width confidence bins.

    Confidence is the probability given to the predicted label. If ``pred`` is None
    the predicted label is the argmax of ``probs``. Passing ``pred`` matters when an
    "Unverified" threshold is applied after calibration (H4), because then the
    prediction is not always the argmax.

    Args:
        gold: true labels, length n.
        probs: (n, 3) probabilities in the order ``LABELS``.
        pred: predicted labels (optional).
        n_bins: number of equal-width bins on [0, 1] (CLAUDE.md: 15).

    Returns:
        ECE = sum over bins of (bin size / n) * |accuracy - mean confidence|,
        or None if empty.
    """
    probs = np.asarray(probs, dtype=float)
    if len(probs) == 0:
        return None
    gold = np.asarray(gold)
    if pred is None:
        pred_idx = probs.argmax(axis=1)
    else:
        pred_idx = np.array([LABELS.index(p) for p in pred])
    conf = probs[np.arange(len(probs)), pred_idx]
    correct = np.array(LABELS)[pred_idx] == gold
    # Bin i covers (i/n_bins, (i+1)/n_bins]; confidence 0 goes into the first bin.
    bin_idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        in_bin = bin_idx == b
        if in_bin.any():
            total += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(total)


# --------------------------------------------------------------------------
# Twin metrics
# --------------------------------------------------------------------------

def evidence_sensitivity(df: pd.DataFrame) -> float | None:
    """ESR: share of ``_B`` (evidence-swap) twins predicted ``contradicted``."""
    return _share(twins(df, ["B"])["pred"] == "contradicted")


def unrelated_rate(df: pd.DataFrame) -> float | None:
    """Share of ``_C`` (unrelated-evidence) twins predicted ``unverified``."""
    return _share(twins(df, ["C"])["pred"] == "unverified")


def paraphrase_consistency(df: pd.DataFrame) -> float | None:
    """Share of ``_P`` twins with the same prediction as their original."""
    return _agreement_with_original(df, ["P"])


def crosslingual_consistency(df: pd.DataFrame,
                             suffixes: Sequence[str] = ("A", "M")) -> float | None:
    """Share of ``_A``/``_M`` twins with the same prediction as the English original.

    Args:
        df: joined table with originals and twins.
        suffixes: which language twins to include; pass ``("A",)`` or ``("M",)``
            for one type only.
    """
    return _agreement_with_original(df, suffixes)


def authenticity_leakage(df: pd.DataFrame) -> float | None:
    """ALR: share of ``_V`` (image) twins whose prediction differs from the original's."""
    agree = _agreement_with_original(df, ["V"])
    return None if agree is None else 1.0 - agree


# --------------------------------------------------------------------------
# Explanation metrics (inputs come from src/explain/faithfulness.py)
# --------------------------------------------------------------------------

def faithfulness(sentence_entailed: Sequence[Sequence[bool]]) -> float | None:
    """Share of explanation sentences entailed by the snippet they cite.

    Args:
        sentence_entailed: one list per explanation, one bool per sentence
            (True = the fine-tuned NLI model says the cited snippet entails it).

    Returns:
        Fraction over all sentences of all explanations, or None if there are none.
    """
    flat = [bool(s) for expl in sentence_entailed for s in expl]
    return float(np.mean(flat)) if flat else None


def citation_accuracy(citation_supports: Sequence[bool]) -> float | None:
    """Share of citations that point to a snippet supporting the cited sentence.

    Args:
        citation_supports: one bool per citation ``[k]`` across all explanations.
    """
    return float(np.mean([bool(c) for c in citation_supports])) if len(citation_supports) else None


# --------------------------------------------------------------------------
# Claim extraction and retrieval
# --------------------------------------------------------------------------

def claim_extraction_f1(pred_claims: Sequence[Sequence[str]],
                        gold_claims: Sequence[Sequence[str]],
                        sim_fn: Callable[[Sequence[str], Sequence[str]], np.ndarray],
                        threshold: float = 0.8) -> dict:
    """Micro precision / recall / F1 of extracted claims against gold claims.

    For each message, predicted and gold claims are matched one-to-one, greedily
    from the most similar pair down; a pair counts as a match if its similarity is
    at least ``threshold``.

    Args:
        pred_claims: one list of predicted claims per message.
        gold_claims: one list of gold claims per message (same order).
        sim_fn: ``sim_fn(preds, golds)`` returns a (len(preds), len(golds)) array of
            similarities; in the project this is BGE-M3 cosine similarity.
        threshold: minimum similarity for a match (CLAUDE.md: 0.8).

    Returns:
        ``{"precision", "recall", "f1", "tp", "n_pred", "n_gold"}``.
    """
    if len(pred_claims) != len(gold_claims):
        raise ValueError("pred_claims and gold_claims must have the same length")
    tp = n_pred = n_gold = 0
    for preds, golds in zip(pred_claims, gold_claims):
        n_pred += len(preds)
        n_gold += len(golds)
        if not preds or not golds:
            continue
        sim = np.asarray(sim_fn(preds, golds), dtype=float)
        used_p, used_g = set(), set()
        # Visit all pairs from most to least similar.
        for flat in np.argsort(-sim, axis=None):
            i, j = np.unravel_index(flat, sim.shape)
            if sim[i, j] < threshold:
                break
            if i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
                tp += 1
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


def recall_at_k(retrieved: Sequence[Sequence], gold: Sequence[Iterable],
                k: int = 5) -> float | None:
    """Share of claims with at least one of their own snippets in the top k.

    Args:
        retrieved: one ranked list of snippet ids per claim (best first).
        gold: one collection of the claim's own snippet ids per claim.
        k: cut-off (CLAUDE.md: 5).

    Returns:
        Fraction of claims with a hit, ignoring claims with no gold snippets;
        None if no claim has gold snippets.
    """
    hits = []
    for ranked, own in zip(retrieved, gold):
        own = set(own)
        if not own:
            continue
        hits.append(bool(own & set(list(ranked)[:k])))
    return float(np.mean(hits)) if hits else None


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame,
                     metric_fn: Callable[[pd.DataFrame], float | None],
                     n_resamples: int = 1000, seed: int = 42) -> dict:
    """Paired bootstrap of the difference metric(A) - metric(B).

    Both models are evaluated on the same resampled items. Items are resampled by
    ``base_id`` (an original together with all its twins), so twin metrics keep
    each twin next to its original.

    Args:
        df_a: joined table for model A.
        df_b: joined table for model B (same ids as A; extra ids are dropped).
        metric_fn: function of a joined table, e.g.
            ``lambda d: macro_f1(originals(d)["gold"], originals(d)["pred"])``
            or ``evidence_sensitivity``.
        n_resamples: number of bootstrap resamples (CLAUDE.md: 1,000).
        seed: random seed (42).

    Returns:
        ``{"a", "b", "diff", "ci_low", "ci_high", "p_value", "n_resamples"}`` where
        the CI is the 95% percentile interval of the difference and the p-value is
        two-sided: 2 * min(share of diffs <= 0, share of diffs >= 0), capped at 1.
    """
    common = sorted(set(df_a["id"]) & set(df_b["id"]))
    a = df_a[df_a["id"].isin(common)].sort_values("id").reset_index(drop=True)
    b = df_b[df_b["id"].isin(common)].sort_values("id").reset_index(drop=True)

    groups = a.groupby("base_id").indices        # base_id -> row positions (same in b)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        sample = rng.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([groups[keys[s]] for s in sample])
        ma, mb = metric_fn(a.iloc[rows]), metric_fn(b.iloc[rows])
        if ma is not None and mb is not None:
            diffs.append(ma - mb)
    diffs = np.array(diffs)
    full_a, full_b = metric_fn(a), metric_fn(b)
    if len(diffs) == 0 or full_a is None or full_b is None:
        return {"a": full_a, "b": full_b, "diff": None, "ci_low": None,
                "ci_high": None, "p_value": None, "n_resamples": 0}
    p_value = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"a": full_a, "b": full_b, "diff": full_a - full_b,
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "p_value": float(p_value), "n_resamples": len(diffs)}


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1) over seeds 42/43/44.

    Returns ``(mean, 0.0)`` for a single value.
    """
    arr = np.asarray(values, dtype=float)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), sd


# --------------------------------------------------------------------------
# Everything at once
# --------------------------------------------------------------------------

def compute_all(df: pd.DataFrame) -> dict:
    """Compute every prediction-based metric that applies to a joined table.

    Args:
        df: joined table from ``src.eval.loaders.join_gold``.

    Returns:
        Dict of metric name -> value (None when that metric has nothing to measure),
        plus counts ``n_orig`` and ``n_<twin type>``.
    """
    orig = originals(df)
    result = {
        "n_orig": len(orig),
        "macro_f1": macro_f1(list(orig["gold"]), list(orig["pred"])),
        "auroc": auroc(orig["gold"], _probs_matrix(orig)),
        "ece": ece(orig["gold"], _probs_matrix(orig), pred=list(orig["pred"])),
        "esr": evidence_sensitivity(df),
        "unrelated_rate": unrelated_rate(df),
        "paraphrase_consistency": paraphrase_consistency(df),
        "crosslingual_consistency": crosslingual_consistency(df),
        "crosslingual_A": crosslingual_consistency(df, ("A",)),
        "crosslingual_M": crosslingual_consistency(df, ("M",)),
        "alr": authenticity_leakage(df),
    }
    for kind in ("P", "A", "M", "B", "C", "V"):
        result[f"n_{kind}"] = int((df["twin_type"] == kind).sum())
    return result
