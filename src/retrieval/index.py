"""Offline evidence search used by the pipeline and the demo (R5).

What it does
    ``search(claim, k)`` returns the ``k`` best snippets for a claim from the offline
    library (BM25 + BGE-M3 candidates, reranked; see ``build_index.py``). The library
    is loaded once, on the first call.

    The library must be on disk first (models/index_v1/, or the folder in the
    TWINCHECK_INDEX environment variable). Get it with
        python -m src.retrieval.build_index download
    If it is missing, importing this module raises ImportError on purpose, so
    ``src/pipeline.py`` (which imports it only if it can) keeps its "no evidence"
    stand-in instead of failing.

Inputs
    claim (str), k (int, default 5).

Outputs
    [{"text", "url", "domain", "language", "record_ids", "score"}, ...]

How to run
    python -m src.retrieval.index "Dubai Metro suspended all lines because of flooding"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

INDEX_DIR = Path(os.environ.get("TWINCHECK_INDEX", "models/index_v1"))
if not (INDEX_DIR / "snippets.jsonl").exists():
    raise ImportError(f"offline index not found in {INDEX_DIR}; run "
                      "python -m src.retrieval.build_index download")

_index = None


def search(claim: str, k: int = 5) -> list[dict]:
    """The ``k`` best snippets for ``claim`` (interface used by src/pipeline.py)."""
    global _index
    if _index is None:
        from src.retrieval.build_index import load
        _index = load(str(INDEX_DIR))
    return _index.search(claim, k)


if __name__ == "__main__":
    for hit in search(" ".join(sys.argv[1:])):
        print(json.dumps({**hit, "text": hit["text"][:120]}, ensure_ascii=False))
