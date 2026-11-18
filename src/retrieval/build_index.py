"""Offline evidence library for the demo: BM25 + BGE-M3 (FAISS) + reranker (R5).

What it does
    build     collects every cleaned snippet of MultiFC, AVeriTeC and X-FACT Arabic
              (all splits; identical texts are stored once, with every claim id that
              used them), embeds them with BAAI/bge-m3 and saves the library.
    search    (``OfflineIndex.search``) for a claim: BM25 top 20 + dense top 20 ->
              the union is scored by the fine-tuned reranker (R2) -> best k.
    eval      recall@5 on 200 English test claims (+ 50 Arabic): does the top 5 hold
              one of the claim's own snippets? Reported for BM25 only, dense only
              and the full hybrid + reranker search.
    download  fetches the saved library from Hugging Face Hub into models/index_v1/
              (the embeddings are ~330 MB, too big for the repo).
    The library is a fixed offline collection (no live web search, CLAUDE.md §3), so
    test claims can find their own snippets: recall@5 measures the search step only.

Inputs
    data/processed/en/{multifc,averitec}_{train,val,test}.jsonl,
    data/processed/ar/xfact_ar_{train,val,test}.jsonl

Outputs
    models/index_v1/{snippets.jsonl, embeddings.npy, meta.json}   (not in git; on the Hub)
    results/R5/retrieval_eval.json

How to run (Colab GPU for build, repo root)
    python -m src.retrieval.build_index build --limit 10 --out-dir models/index_test10   # 10-row check
    python -m src.retrieval.build_index build
    python -m src.retrieval.build_index eval --n-en 200 --n-ar 50
    python -m src.retrieval.build_index download --repo RathiAbeth/twincheck-index-v1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np

EN = "data/processed/en"
AR = "data/processed/ar"
SOURCE_FILES = ([f"{EN}/{d}_{s}.jsonl" for d in ("multifc", "averitec")
                 for s in ("train", "val", "test")]
                + [f"{AR}/xfact_ar_{s}.jsonl" for s in ("train", "val", "test")])
INDEX_DIR = "models/index_v1"
HUB_REPO = "RathiAbeth/twincheck-index-v1"
ENCODER = "BAAI/bge-m3"
RERANKER = "RathiAbeth/twincheck-reranker-v1"
N_CANDIDATES = 20
WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens (\\w+ also matches Arabic letters)."""
    return WORD_RE.findall(text.lower())


def collect_snippets(records: list[dict]) -> list[dict]:
    """Unique snippets with every claim id that used them.

    Returns:
        [{"text", "url", "domain", "language", "record_ids": [...]}] in first-seen order.
    """
    by_text: dict[str, dict] = {}
    for rec in records:
        for ev in rec["evidence"]:
            text = ev["text"].strip()
            if not text:
                continue
            if text not in by_text:
                by_text[text] = {"text": text, "url": ev.get("url"), "domain": ev.get("domain"),
                                 "language": rec.get("language", "en"), "record_ids": []}
            if rec["id"] not in by_text[text]["record_ids"]:
                by_text[text]["record_ids"].append(rec["id"])
    return list(by_text.values())


class OfflineIndex:
    """Hybrid search over a fixed snippet library.

    Args:
        snippets: output of ``collect_snippets``.
        embeddings: (n, d) array of L2-normalised snippet vectors (same order).
        encode: function(list[str]) -> (m, d) normalised query vectors.
        rerank: function(claim, list[str]) -> list of relevance scores.
    """

    def __init__(self, snippets: list[dict], embeddings: np.ndarray, encode, rerank):
        """Build the BM25 index and the FAISS inner-product index over ``snippets``."""
        from rank_bm25 import BM25Okapi
        import faiss

        self.snippets = snippets
        self.encode = encode
        self.rerank = rerank
        self.bm25 = BM25Okapi([tokenize(s["text"]) or ["_"] for s in snippets])
        self.faiss = faiss.IndexFlatIP(embeddings.shape[1])   # inner product = cosine here
        self.faiss.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    def bm25_top(self, claim: str, n: int = N_CANDIDATES) -> list[int]:
        """Indices of the ``n`` best BM25 matches."""
        scores = self.bm25.get_scores(tokenize(claim))
        return [int(i) for i in np.argsort(-scores)[:n]]

    def dense_top(self, claim: str, n: int = N_CANDIDATES) -> list[int]:
        """Indices of the ``n`` nearest snippets by embedding."""
        query = np.asarray(self.encode([claim]), dtype=np.float32)
        _, idx = self.faiss.search(query, n)
        return [int(i) for i in idx[0] if i >= 0]

    def search(self, claim: str, k: int = 5, mode: str = "hybrid") -> list[dict]:
        """Best ``k`` snippets for a claim.

        Args:
            mode: "hybrid" (BM25 + dense candidates, reranked), "bm25" or "dense".

        Returns:
            Snippet dicts (text, url, domain, language, record_ids) with a "score".
        """
        if mode == "bm25":
            return [dict(self.snippets[i], score=None) for i in self.bm25_top(claim, k)]
        if mode == "dense":
            return [dict(self.snippets[i], score=None) for i in self.dense_top(claim, k)]
        candidates = list(dict.fromkeys(self.bm25_top(claim) + self.dense_top(claim)))
        scores = self.rerank(claim, [self.snippets[i]["text"] for i in candidates])
        best = sorted(zip(candidates, scores), key=lambda p: -p[1])[:k]
        return [dict(self.snippets[i], score=round(float(s), 4)) for i, s in best]


def bge_encoder(batch_size: int = 64):
    """BGE-M3 encode function (fp16 on GPU), returning normalised vectors."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(ENCODER, device=device)
    model.max_seq_length = 256
    if device == "cuda":
        model.half()

    def encode(texts: list[str]) -> np.ndarray:
        """Unit-length BGE-M3 embeddings, one row per text."""
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=len(texts) > 1000)
    return encode


def reranker_fn(model: str = RERANKER):
    """Relevance function from the fine-tuned reranker (R2)."""
    from src.retrieval.rerank import Reranker

    return Reranker(model).score


def save(out_dir: str, snippets: list[dict], embeddings: np.ndarray, meta: dict) -> None:
    """Write snippets.jsonl, embeddings.npy (float16) and meta.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "snippets.jsonl", "w", encoding="utf-8") as f:
        for s in snippets:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    np.save(out / "embeddings.npy", embeddings.astype(np.float16))
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load(index_dir: str = INDEX_DIR, encode=None, rerank=None) -> OfflineIndex:
    """Load a saved library; the BGE-M3 encoder and reranker load unless given."""
    from src.eval.loaders import load_jsonl

    snippets = load_jsonl(Path(index_dir) / "snippets.jsonl")
    embeddings = np.load(Path(index_dir) / "embeddings.npy")
    return OfflineIndex(snippets, embeddings, encode or bge_encoder(), rerank or reranker_fn())


def recall_at_k(index: OfflineIndex, records: list[dict], k: int = 5,
                mode: str = "hybrid") -> float | None:
    """Share of claims whose top ``k`` results include one of their own snippets."""
    hits = [any(rec["id"] in s["record_ids"] for s in index.search(rec["claim"], k, mode))
            for rec in records]
    return sum(hits) / len(hits) if hits else None


def sample_records(paths: list[str], n: int, seed: int = 42) -> list[dict]:
    """``n`` records with snippets, sorted by id and shuffled with ``seed``."""
    from src.eval.loaders import load_jsonl

    recs = sorted((r for p in paths if Path(p).exists() for r in load_jsonl(p)
                   if any(ev["text"].strip() for ev in r["evidence"])), key=lambda r: r["id"])
    random.Random(seed).shuffle(recs)
    return recs[:n]


def main() -> None:
    """CLI: build, eval or download (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["build", "eval", "download"])
    ap.add_argument("--files", nargs="+", default=SOURCE_FILES)
    ap.add_argument("--out-dir", default=INDEX_DIR)
    ap.add_argument("--limit", type=int, help="first N records per file (10-row check)")
    ap.add_argument("--n-en", type=int, default=200)
    ap.add_argument("--n-ar", type=int, default=50)
    ap.add_argument("--reranker", default=RERANKER)
    ap.add_argument("--repo", default=HUB_REPO)
    ap.add_argument("--report", default="results/R5/retrieval_eval.json")
    args = ap.parse_args()

    if args.command == "download":
        from huggingface_hub import snapshot_download
        snapshot_download(args.repo, repo_type="dataset", local_dir=args.out_dir)
        print(f"downloaded {args.repo} -> {args.out_dir}")
        return

    if args.command == "build":
        from src.eval.loaders import load_jsonl
        records = []
        for p in args.files:
            rows = load_jsonl(p)
            records += rows[:args.limit] if args.limit else rows
        snippets = collect_snippets(records)
        start = time.time()
        embeddings = bge_encoder()([s["text"] for s in snippets])
        meta = {"encoder": ENCODER, "n_records": len(records), "n_snippets": len(snippets),
                "files": args.files, "embed_minutes": round((time.time() - start) / 60, 1)}
        save(args.out_dir, snippets, embeddings, meta)
        print(json.dumps(meta, indent=2))
        return

    index = load(args.out_dir, rerank=reranker_fn(args.reranker))
    en = sample_records([f"{EN}/multifc_test.jsonl", f"{EN}/averitec_test.jsonl"], args.n_en)
    ar = sample_records([f"{AR}/xfact_ar_test.jsonl"], args.n_ar)
    report = {"index_dir": args.out_dir, "n_snippets": len(index.snippets), "k": 5,
              "candidates_per_method": N_CANDIDATES}
    for name, recs in (("en", en), ("ar", ar)):
        report[name] = {"n_claims": len(recs),
                        **{f"recall@5_{m}": recall_at_k(index, recs, 5, m)
                           for m in ("bm25", "dense", "hybrid")}}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
