"""Live web evidence for the demo: public search results, fact-check sites removed, reranked (R10).

What it does
    ``search(claim, k=5)`` asks DuckDuckGo (no API key; web and news results) for the
    claim, keeps each result's title + snippet as one evidence snippet in the shared
    format ({"text", "url", "domain"}), drops results from fact-checking sites (they state
    the verdict; same list and rule as src/data/clean.py), and keeps the ``k`` snippets the
    fine-tuned reranker (R2) finds most relevant, with their score as "relevance".
    Demo only (Team Brief: "live search is for the demo only"); the experiments use the
    offline library (src/retrieval/index.py). The claim text is sent to DuckDuckGo.
    If the search fails (no internet, rate limit), it returns [] and ``last_error`` says why.

Inputs
    claim (str); optional reranker name (default RathiAbeth/twincheck-reranker-v1).

Outputs
    list of {"text", "url", "domain", "relevance"} (best first).

How to run (repo root; pip install ddgs)
    python -m src.retrieval.live "Gold rates in Dubai: 24K AED 437.75"
"""

from __future__ import annotations

import argparse
import json
from urllib.parse import urlparse

from src.data.clean import FACTCHECK_DOMAINS, is_factcheck_site

RERANKER = "RathiAbeth/twincheck-reranker-v1"
N_RESULTS = 10            # per search type (web, news) before filtering and reranking
last_error: str | None = None
_reranker = None


def domain(url: str) -> str:
    """'www.khaleejtimes.com' -> 'khaleejtimes.com'."""
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def web_results(claim: str, n: int = N_RESULTS) -> list[dict]:
    """Raw DuckDuckGo web + news results as snippets, duplicates (same URL) removed."""
    from ddgs import DDGS

    snippets, seen = [], set()
    with DDGS() as ddgs:
        hits = list(ddgs.text(claim, max_results=n))
        try:
            hits += [{"title": h.get("title", ""), "body": h.get("body", ""), "href": h.get("url", "")}
                     for h in ddgs.news(claim, max_results=n)]
        except Exception:                      # news search is a bonus; web results are enough
            pass
    for h in hits:
        url, text = h.get("href", ""), " ".join(x for x in (h.get("title"), h.get("body")) if x).strip()
        if url and text and url not in seen:
            seen.add(url)
            snippets.append({"text": text, "url": url, "domain": domain(url)})
    return snippets


def rerank(claim: str, snippets: list[dict], k: int, model: str = RERANKER) -> list[dict]:
    """The ``k`` snippets the reranker scores highest (each gets "relevance")."""
    global _reranker
    if not snippets:
        return []
    try:
        if _reranker is None:
            from src.retrieval.rerank import Reranker

            _reranker = Reranker(model)
        scores = _reranker.score(claim, [s["text"] for s in snippets])
    except Exception:                          # no torch / model: keep the search order
        return snippets[:k]
    ranked = sorted(zip(scores, snippets), key=lambda p: -p[0])[:k]
    return [{**s, "relevance": round(float(score), 4)} for score, s in ranked]


def search(claim: str, k: int = 5, model: str = RERANKER) -> list[dict]:
    """Live evidence for one claim: web search -> drop fact-check sites -> top ``k`` by reranker."""
    global last_error
    last_error = None
    try:
        found = web_results(claim)
    except Exception as err:
        last_error = f"live search failed ({type(err).__name__}: {err})"
        return []
    kept = [s for s in found if not is_factcheck_site(s, FACTCHECK_DOMAINS)]
    if not kept:
        last_error = "the web search found no usable results"
    return rerank(claim, kept, k, model)


def main() -> None:
    """CLI: print the live evidence for one claim."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("claim")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    print(json.dumps(search(args.claim, args.k), indent=2, ensure_ascii=False))
    if last_error:
        print("note:", last_error)


if __name__ == "__main__":
    main()
