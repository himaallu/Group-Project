"""Unit tests for live web evidence (R10). No internet: the search is replaced by fixed results.

Run: python -m pytest tests/retrieval/test_live.py -q
"""

from src.retrieval import live

HITS = [{"text": "Gold rate today: 24K AED 437.75 per gram.", "url": "https://www.goldtrack.ae/dubai", "domain": "goldtrack.ae"},
        {"text": "Fact check: gold rumour is false.", "url": "https://www.snopes.com/x", "domain": "snopes.com"},
        {"text": "Gold fact check", "url": "https://news.example.com/factcheck-gold", "domain": "news.example.com"},
        {"text": "Dubai gold souk opening hours.", "url": "https://visitdubai.com/souk", "domain": "visitdubai.com"}]


def test_fact_check_sites_removed_and_top_k(monkeypatch):
    monkeypatch.setattr(live, "web_results", lambda claim, n=10: list(HITS))
    monkeypatch.setattr(live, "rerank", lambda claim, s, k, model=None: s[:k])
    out = live.search("Gold rates in Dubai 24K AED 437.75", k=5)
    assert [s["domain"] for s in out] == ["goldtrack.ae", "visitdubai.com"]   # snopes + /factcheck- dropped
    assert live.last_error is None


def test_search_failure_returns_empty_with_reason(monkeypatch):
    def boom(claim, n=10):
        raise ConnectionError("no internet")
    monkeypatch.setattr(live, "web_results", boom)
    assert live.search("anything") == [] and "no internet" in live.last_error


def test_domain():
    assert live.domain("https://www.khaleejtimes.com/gold") == "khaleejtimes.com"


def test_demo_uses_live_evidence(monkeypatch):
    from src.demo.app import WEB, run_check

    monkeypatch.setenv("TWINCHECK_EXPLAINER", "template")
    monkeypatch.setenv("TWINCHECK_EXTRACTOR", "split")
    monkeypatch.setattr(live, "search", lambda claim, k=5: [HITS[0]])
    out = run_check("Gold rates in Dubai: 24K AED 437.75.", "", "en", source=WEB)
    assert "Evidence retrieved from the web" in out and "goldtrack.ae" in out and "Offline evidence library" not in out


def test_keywords_keep_numbers():
    q = live.keywords("Today's retail gold rate in Dubai is AED 437.75 per gram for 24K gold.")
    assert "437.75" in q and "24K" in q and "is" not in q.split()


def test_falls_back_to_keywords_when_full_claim_fails(monkeypatch):
    calls = []

    def engine(query, n):
        calls.append(query)
        if len(calls) == 1:
            raise RuntimeError("no search engine answered")
        return [{"title": "Gold", "body": "24K AED 437.75", "href": "https://goldsouk.ae/rates"}]
    monkeypatch.setattr(live, "_engine_hits", engine)
    out = live.web_results("Today's retail gold rate in Dubai is AED 437.75 per gram.")
    assert len(calls) == 2 and calls[1] == live.keywords(calls[0]) and out[0]["domain"] == "goldsouk.ae"


def test_cache_answers_without_the_network(monkeypatch, tmp_path):
    monkeypatch.setattr(live, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(live, "_cache", None)
    monkeypatch.setattr(live, "web_results", lambda claim, n=10: [HITS[0]])
    monkeypatch.setattr(live, "rerank", lambda claim, s, k, model=None: s[:k])
    assert live.warm(["gold claim"]) == {"gold claim": 1}

    def offline(claim, n=10):
        raise ConnectionError("no internet")
    monkeypatch.setattr(live, "web_results", offline)
    monkeypatch.setattr(live, "_cache", None)                     # a fresh start reads the file
    assert live.search("gold claim")[0]["domain"] == "goldtrack.ae" and live.last_error is None


def test_demo_falls_back_to_library_when_search_fails(monkeypatch):
    from src.demo.app import WEB, run_check

    monkeypatch.setenv("TWINCHECK_EXPLAINER", "template")
    monkeypatch.setenv("TWINCHECK_EXTRACTOR", "split")

    def failed(claim, k=5):
        live.last_error = "web search unavailable (no search engine answered)"
        return []
    monkeypatch.setattr(live, "search", failed)
    out = run_check("Gold rates in Dubai: 24K AED 437.75.", "", "en", source=WEB)
    assert "Web search was unavailable, so evidence was taken from the offline library" in out
    assert "DDGSException" not in out
