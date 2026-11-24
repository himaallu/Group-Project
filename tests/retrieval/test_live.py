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
    assert "live web search" in out and "goldtrack.ae" in out and "Offline evidence library" not in out
