# src/retrieval: relevance and evidence search

| File | Owner | What it does |
|---|---|---|
| `build_pairs.py` | Rathi (R2) | (claim, snippet) relevance pairs for the reranker |
| `train_reranker.py` | Rathi (R2) | fine-tunes `BAAI/bge-reranker-v2-m3` (W&B, seed 42) |
| `rerank.py` | Rathi (R2) | `Reranker(model).score(claim, snippets)`; scores files → `features/relevance_*.parquet` |
| `build_index.py` | Rathi (R5) | offline library: BM25 + BGE-M3 (FAISS) + reranker; build / eval (recall@5) / download |
| `index.py` | Rathi (R5) | `search(claim, k)` used by `src/pipeline.py` (needs the library on disk) |

Models and the library are on the Hugging Face Hub (see `docs/implementation.md` §7). Tests: `tests/retrieval/`, `tests/explain/`.
