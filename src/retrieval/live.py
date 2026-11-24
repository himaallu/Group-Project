"""Live web evidence for the demo: public search results, fact-check sites removed, reranked (R10).

What it does
    ``search(claim, k=5)`` asks DuckDuckGo (no API key; web and news results) for the
    claim, keeps each result's title + snippet as one evidence snippet in the shared
    format ({"text", "url", "domain"}), drops results from fact-checking sites (they state
    the verdict; same list and rule as src/data/clean.py), and keeps the ``k`` snippets the
    fine-tuned reranker (R2) finds most relevant, with their score as "relevance".
    Demo only (Team Brief: "live search is for the demo only"); the experiments use the
    offline library (src/retrieval/index.py). The claim text is sent to the search engines.
    Search engines often refuse automated requests for a while, so ``search`` is robust:
      1. it tries several engines through the ddgs package (Bing, Yahoo, auto, DuckDuckGo,
         Brave, Mojeek) until one returns results, and repeats once after a short pause;
      2. if the full claim finds nothing, it retries with a shorter keyword query;
      3. every successful search is cached in memory and in a JSON file (TWINCHECK_LIVE_CACHE,
         default: twincheck_live_cache.json in the system temp folder), and ``warm(claims)`` fills the cache before a demo.
    If everything fails, it returns [] and ``last_error`` says why; the demo then falls
    back to the offline library.

Inputs
    claim (str); optional reranker name (default RathiAbeth/twincheck-reranker-v1).

Outputs
    list of {"text", "url", "domain", "relevance"} (best first).

How to run (repo root; pip install ddgs)
    python -m src.retrieval.live "Gold rates in Dubai: 24K AED 437.75"
    python -m src.retrieval.live --warm "claim one" "claim two"     # fill the cache before a demo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from src.data.clean import FACTCHECK_DOMAINS, is_factcheck_site

RERANKER = "RathiAbeth/twincheck-reranker-v1"
N_RESULTS = 10            # per search type (web, news) before filtering and reranking
BACKENDS = ("bing", "yahoo", "auto", "duckduckgo", "brave", "mojeek")
CACHE_PATH = Path(os.environ.get("TWINCHECK_LIVE_CACHE",
                               Path(tempfile.gettempdir()) / "twincheck_live_cache.json"))
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "for", "and", "to", "at",
             "by", "with", "that", "this", "it", "as", "be", "has", "have", "today", "today's"}
last_error: str | None = None
_reranker = None
_cache: dict | None = None


def domain(url: str) -> str:
    """'www.khaleejtimes.com' -> 'khaleejtimes.com'."""
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def keywords(claim: str, n: int = 10) -> str:
    """A shorter query: the first ``n`` content words of the claim, numbers kept."""
    words = [w for w in re.findall(r"[\w.'%-]+", claim) if w.lower() not in STOPWORDS]
    return " ".join(words[:n])


def _engine_hits(query: str, n: int) -> list[dict]:
    """Text results from the first engine that answers (two rounds); raises if all fail."""
    from ddgs import DDGS

    errors = []
    for _ in range(2):
        for backend in BACKENDS:
            try:
                hits = DDGS().text(query, max_results=n, backend=backend)
                if hits:
                    return list(hits)
            except Exception as err:          # this engine refused or found nothing: try the next
                errors.append(f"{backend}: {type(err).__name__}")
        time.sleep(1.5)
    raise RuntimeError("no search engine answered (" + ", ".join(errors[-3:]) + ")")


def web_results(claim: str, n: int = N_RESULTS) -> list[dict]:
    """Raw web (+ news, if available) results as snippets, duplicates (same URL) removed.

    Tries the full claim first, then a shorter keyword query.
    """
    from ddgs import DDGS

    snippets, seen = [], set()
    try:
        hits = _engine_hits(claim, n)
    except RuntimeError:
        hits = _engine_hits(keywords(claim), n)
    try:
        hits += [{"title": h.get("title", ""), "body": h.get("body", ""), "href": h.get("url", "")}
                 for h in DDGS().news(claim, max_results=n)]
    except Exception:                          # news search is a bonus; web results are enough
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


def _load_cache() -> dict:
    """claim -> snippets saved by earlier searches (memory first, then the JSON file)."""
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _save(claim: str, snippets: list[dict]) -> None:
    """Remember a successful search (memory and file; a write failure is ignored)."""
    cache = _load_cache()
    cache[claim] = snippets
    try:
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def search(claim: str, k: int = 5, model: str = RERANKER) -> list[dict]:
    """Live evidence for one claim: cache, else web search -> drop fact-check sites -> top ``k``."""
    global last_error
    last_error = None
    cached = _load_cache().get(claim)
    if cached:
        return cached[:k]
    try:
        found = web_results(claim)
    except Exception as err:
        last_error = f"web search unavailable ({err})"
        return []
    kept = [s for s in found if not is_factcheck_site(s, FACTCHECK_DOMAINS)]
    if not kept:
        last_error = "the web search found no usable results"
        return []
    best = rerank(claim, kept, k, model)
    _save(claim, best)
    return best


def warm(claims: list[str], k: int = 5) -> dict:
    """Search each claim now and keep the results in the cache (run before a demo)."""
    return {c: len(search(c, k)) for c in claims}


def main() -> None:
    """CLI: print the live evidence for one claim."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("claim", nargs="?")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--warm", nargs="+", metavar="CLAIM", help="search these claims now and cache them")
    args = ap.parse_args()
    if args.warm:
        for claim, n in warm(args.warm, args.k).items():
            print(f"{n} snippets cached: {claim[:70]}")
        return
    print(json.dumps(search(args.claim, args.k), indent=2, ensure_ascii=False))
    if last_error:
        print("note:", last_error)


if __name__ == "__main__":
    main()
