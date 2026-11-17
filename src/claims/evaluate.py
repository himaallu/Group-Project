"""Evaluate claim extraction against the gold examples (task Y3, part 2).

What it does
    Runs src.claims.extract on every gold example except the 6 few-shot ones,
    then computes claim-extraction precision / recall / F1 per language (en, ar,
    mixed) and overall with src.eval.metrics.claim_extraction_f1. Two claims match
    when their BGE-M3 cosine similarity is >= 0.8 (CLAUDE.md section 5).
    Logs the numbers to W&B (project twincheck) unless --no_wandb.

Inputs
    --gold      data/claims/gold.jsonl (falls back to the draft file)
    --backend   qwen | split  (split = rule-based baseline, runs on CPU)

Outputs
    results/Y3/preds_<backend>.jsonl   {"id", "language", "pred_claims", "gold_claims"}
    results/Y3/f1_<backend>.json       scores per language + overall

How to run (repo root)
    python -m src.claims.evaluate --backend split --limit 10 --no_wandb   # quick check
    python -m src.claims.evaluate --backend qwen                          # GPU
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.claims import extract
from src.data.common import SEED, read_jsonl, write_jsonl
from src.eval.metrics import claim_extraction_f1

_encoder = None


def bge_sim(preds, golds) -> np.ndarray:
    """Cosine similarity matrix with BGE-M3 (multilingual, so Arabic and English both work)."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("BAAI/bge-m3")
    a = _encoder.encode(list(preds), normalize_embeddings=True)
    b = _encoder.encode(list(golds), normalize_embeddings=True)
    return a @ b.T


def main() -> None:
    """Predict, score per language, save files and (optionally) log to W&B."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=None)
    ap.add_argument("--backend", choices=["qwen", "split"], default="qwen")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    os.environ["TWINCHECK_EXTRACTOR"] = args.backend
    gold_path = args.gold or next(p for p in extract.GOLD_FILES if os.path.exists(p))
    rows = [r for r in read_jsonl(gold_path) if r["id"] not in extract.FEWSHOT_IDS][: args.limit]

    preds = []
    for i in range(0, len(rows), args.batch):
        preds += extract.extract_batch([r["message"] for r in rows[i:i + args.batch]])
        print(f"{min(i + args.batch, len(rows))}/{len(rows)}")
    out_rows = [{"id": r["id"], "language": r["language"], "pred_claims": p, "gold_claims": r["claims"]}
                for r, p in zip(rows, preds)]

    scores = {}
    for lang in ("en", "ar", "mixed", "all"):
        sub = [o for o in out_rows if lang == "all" or o["language"] == lang]
        if sub:
            scores[lang] = claim_extraction_f1([o["pred_claims"] for o in sub],
                                               [o["gold_claims"] for o in sub], bge_sim)
            scores[lang]["n_messages"] = len(sub)
    os.makedirs("results/Y3", exist_ok=True)
    write_jsonl(out_rows, f"results/Y3/preds_{args.backend}.jsonl")
    json.dump(scores, open(f"results/Y3/f1_{args.backend}.json", "w"), indent=2)
    for lang, s in scores.items():
        print(f"{lang:6s} n={s['n_messages']:3d}  P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")

    if not args.no_wandb:
        import wandb
        run = wandb.init(project="twincheck", name=f"Y3-extract-{args.backend}",
                         config={"task": "Y3", "backend": args.backend, "model": extract.MODEL,
                                 "fewshot": extract.FEWSHOT_IDS, "seed": SEED, "gold": gold_path})
        run.log({f"{lang}/{k}": v for lang, s in scores.items() for k, v in s.items()})
        run.finish()


if __name__ == "__main__":
    main()
