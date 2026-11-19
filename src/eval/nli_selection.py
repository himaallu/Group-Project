"""Why mDeBERTa-v3-base for NLI? Compare zero-shot NLI checkpoints on the gold pairs (H7).

What it does
    Scores the 300 hand-labelled gold pairs (``data/nli/gold_val.jsonl``, 150 EN +
    150 AR) with each candidate checkpoint (premise = snippet, hypothesis = claim, as
    in the pipeline) and records, per language: macro-F1 with a bootstrap 95% CI
    (1,000 resamples, seed 42), accuracy, how often "neutral" is predicted, the number
    of parameters and the CPU time per pair. English-only checkpoints are scored on EN
    only. Fine-tuned v1 / v2 (R4) are read from their saved feature files.

Candidates
    mdeberta_base    MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (ours)
    xlmr_large       joeddav/xlm-roberta-large-xnli          (multilingual, 2x larger)
    deberta_large_en MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (English only)

Outputs
    results/H7/nli_selection.csv

How to run (CPU, ~10 min including downloads)
    python -m src.eval.nli_selection
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.nli.gold_check import NLI_LABELS, load_scores
from src.nli.score_zeroshot import load_model, score_pairs

GOLD = "data/nli/gold_val.jsonl"
OUT = Path("results/H7/nli_selection.csv")
CANDIDATES = {
    "mdeberta_base": ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", ("en", "ar")),
    "xlmr_large": ("joeddav/xlm-roberta-large-xnli", ("en", "ar")),
    "deberta_large_en": ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", ("en",)),
}
FINE_TUNED = {"finetuned_v1": "features/nli_v1", "finetuned_v2": "features/nli_v2"}


def f1_ci(y_true: list[str], y_pred: list[str], n: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Macro-F1 and its bootstrap 95% percentile interval."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    f1 = lambda t, p: f1_score(t, p, labels=list(NLI_LABELS), average="macro", zero_division=0)
    rng = np.random.default_rng(seed)
    boots = [f1(y_true[i], y_pred[i]) for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return f1(y_true, y_pred), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarise(name: str, lang: str, gold: list[str], pred: list[str], params: float | None,
              ms: float | None) -> dict:
    """One output row."""
    f1, lo, hi = f1_ci(gold, pred)
    return {"model": name, "language": lang, "n": len(gold), "macro_f1": round(f1, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "accuracy": round(float(np.mean(np.array(gold) == np.array(pred))), 4),
            "pred_neutral_share": round(pred.count("neutral") / len(pred), 4),
            "gold_neutral_share": round(gold.count("neutral") / len(gold), 4),
            "params_millions": params, "cpu_ms_per_pair": ms}


def main() -> None:
    """Score every candidate and write the table."""
    gold = [json.loads(line) for line in open(GOLD, encoding="utf-8") if line.strip()]
    rows = []
    for name, (ckpt, langs) in CANDIDATES.items():
        tok, model, device = load_model(ckpt, device="cpu")
        params = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
        for lang in langs:
            pairs = [g for g in gold if g["language"] == lang]
            # warm-up pass so the timing below does not include one-off start-up costs
            score_pairs([g["snippet"] for g in pairs[:16]], [g["claim"] for g in pairs[:16]],
                        tok, model, device, batch_size=16)
            start = time.time()
            probs = score_pairs([g["snippet"] for g in pairs], [g["claim"] for g in pairs],
                                tok, model, device, batch_size=16)
            ms = round(1000 * (time.time() - start) / len(pairs), 1)
            pred = [NLI_LABELS[int(np.argmax(r))] for r in probs[["p_entail", "p_contradict", "p_neutral"]].to_numpy()]
            rows.append(summarise(name, lang, [g["label"] for g in pairs], pred, params, ms))
            print(rows[-1], flush=True)
        del model
    for name, folder in FINE_TUNED.items():
        if not Path(folder).exists():
            continue
        scores = load_scores(folder)
        for lang in ("en", "ar"):
            found = [(g["label"], scores[(g["claim_id"], g["snippet_idx"])]) for g in gold
                     if g["language"] == lang and (g["claim_id"], g["snippet_idx"]) in scores]
            if found:
                pred = [NLI_LABELS[int(np.argmax(p))] for _, p in found]
                rows.append(summarise(name, lang, [t for t, _ in found], pred, 278.8, None))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
