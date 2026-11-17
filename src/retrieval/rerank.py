"""Relevance scores for (claim, snippet) pairs with the BGE reranker (R2).

What it does
    ``Reranker`` wraps a cross-encoder (``BAAI/bge-reranker-v2-m3`` or our fine-tuned
    ``models/reranker_v1``). ``score(claim, snippets)`` gives one relevance in [0, 1]
    per snippet (sigmoid of the model's single logit). It is reused by R3 (twin
    sources, unrelated evidence), R5 (offline index) and the pipeline.
    The CLI scores every snippet of every record in the given files.

Inputs
    JSONL files in the shared record format, e.g. data/processed/en/*_{train,val,test}.jsonl.

Outputs
    One parquet file with columns  id, snippet_idx, relevance  (one row per snippet;
    snippet_idx is the position in the record's ``evidence`` list).

How to run (Colab GPU)
    # 10-row check first
    python -m src.retrieval.rerank data/processed/en/multifc_val.jsonl --limit 10 \
        --model models/reranker_v1 --out features/test10/relevance_en.parquet
    # full run (~20 min on a T4 for all English splits)
    python -m src.retrieval.rerank data/processed/en/{multifc,averitec}_{train,val,test}.jsonl \
        --model models/reranker_v1 --out features/relevance_en.parquet

    In Python:
        from src.retrieval.rerank import Reranker
        rr = Reranker("models/reranker_v1")
        rr.score("Dubai Metro suspended all lines", ["Green Line operating normally."])
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.eval.loaders import load_jsonl

BASE_MODEL = "BAAI/bge-reranker-v2-m3"
FINETUNED_MODEL = "models/reranker_v1"
OUT_COLUMNS = ("id", "snippet_idx", "relevance")


def length_order(claims: list[str], snippets: list[str]) -> list[int]:
    """Indices of the pairs sorted by total length (short first).

    Batches of similar length need less padding, so scoring is faster. Results are
    put back in the input order afterwards.
    """
    return sorted(range(len(claims)), key=lambda i: len(claims[i]) + len(snippets[i]))


class Reranker:
    """A cross-encoder that scores how relevant a snippet is to a claim."""

    def __init__(self, model_name: str = FINETUNED_MODEL, device: str | None = None,
                 max_len: int = 256, batch_size: int = 64):
        """Load tokenizer and model (fp16 on GPU) in eval mode.

        Args:
            model_name: local folder or Hugging Face id.
            device: "cuda" or "cpu" (default: cuda if available).
            max_len: max tokens per (claim, snippet) pair; longer pairs are cut.
            batch_size: pairs per forward pass.
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()  # fp16 is ~2x faster on a T4; scores agree to ~3 decimals
        self.max_len = max_len
        self.batch_size = batch_size

    def score_pairs(self, claims: list[str], snippets: list[str]) -> list[float]:
        """Relevance in [0, 1] for aligned lists of claims and snippets.

        Args:
            claims: claim texts.
            snippets: snippet texts (same length as ``claims``).

        Returns:
            One float per pair, in the input order.
        """
        import torch

        order = length_order(claims, snippets)
        scores = [0.0] * len(claims)
        with torch.no_grad():
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                enc = self.tokenizer([claims[i] for i in batch], [snippets[i] for i in batch],
                                     truncation=True, max_length=self.max_len,
                                     padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits[:, 0].float()
                for i, value in zip(batch, torch.sigmoid(logits).cpu().tolist()):
                    scores[i] = value
        return scores

    def score(self, claim: str, snippets: list[str]) -> list[float]:
        """Relevance of each snippet to one claim (empty list in, empty list out)."""
        return self.score_pairs([claim] * len(snippets), snippets) if snippets else []


def snippet_rows(records: list[dict]) -> list[dict]:
    """One row per non-empty snippet: id, snippet_idx, claim, snippet."""
    rows = []
    for rec in records:
        for idx, ev in enumerate(rec.get("evidence") or []):
            text = (ev.get("text") or "").strip()
            if text:
                rows.append({"id": rec["id"], "snippet_idx": idx,
                             "claim": rec["claim"], "snippet": text})
    return rows


def score_files(paths: list[str], reranker: Reranker, limit: int | None = None):
    """Score every snippet in the given files.

    Args:
        paths: shared-format JSONL files.
        reranker: a loaded ``Reranker``.
        limit: first N records per file only (10-row check).

    Returns:
        pandas DataFrame with columns id, snippet_idx, relevance.
    """
    import pandas as pd

    rows = []
    for path in paths:
        records = load_jsonl(path)
        records = records[:limit] if limit else records
        file_rows = snippet_rows(records)
        start = time.time()
        scores = reranker.score_pairs([r["claim"] for r in file_rows],
                                      [r["snippet"] for r in file_rows])
        for row, value in zip(file_rows, scores):
            rows.append({"id": row["id"], "snippet_idx": row["snippet_idx"], "relevance": value})
        print(f"{path}: {len(records)} records, {len(file_rows)} snippets "
              f"({time.time() - start:.0f}s on {reranker.device})")
    return pd.DataFrame(rows, columns=list(OUT_COLUMNS))


def main() -> None:
    """CLI: score files and write one parquet file."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="shared-format JSONL files")
    ap.add_argument("--model", default=FINETUNED_MODEL)
    ap.add_argument("--out", default="features/relevance_en.parquet")
    ap.add_argument("--limit", type=int, help="first N records per file (10-row check)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    reranker = Reranker(args.model, max_len=args.max_len, batch_size=args.batch_size)
    table = score_files(args.files, reranker, args.limit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(f"{len(table)} rows -> {args.out}; mean relevance {table['relevance'].mean():.3f}")


if __name__ == "__main__":
    main()
