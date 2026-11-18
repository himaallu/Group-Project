"""Unit tests for the R6 demo (no Gradio, GPU or model download needed).

Run: python -m pytest tests/demo -q
"""

from collections import Counter

from src.demo.app import ILLUSTRATIVE, claim_card, format_results, parse_evidence, run_check
from src.demo.examples import EXAMPLES
from src.pipeline import detect_language


def test_examples_cover_every_verdict_and_language():
    assert len(EXAMPLES) == 6
    assert Counter(e["expected"] for e in EXAMPLES) == {"supported": 2, "contradicted": 2,
                                                        "unverified": 2}
    langs = Counter(e["lang"] for e in EXAMPLES)
    assert langs["ar"] >= 2 and langs["mixed"] >= 1
    for e in EXAMPLES:                       # the declared language matches the text
        assert detect_language(e["message"]) == e["lang"], e["title"]


def test_parse_evidence():
    assert parse_evidence("  \n") is None
    assert parse_evidence("a\n\n b \n") == ["a", "b"]


def test_card_escapes_html_and_shows_confidence():
    result = {"claim": "<b>x</b>", "label": "contradicted", "confidence": 0.734,
              "explanation": "Trains ran [1].",
              "evidence": [{"text": "Trains <ran>", "url": "https://a.ae", "domain": "a.ae"}]}
    card = claim_card(result)
    assert "&lt;b&gt;x&lt;/b&gt;" in card and "Contradicted" in card and "73%" in card
    assert 'href="https://a.ae"' in card and 'dir="auto"' in card
    assert "No check-worthy claim" in format_results([])
    assert ILLUSTRATIVE in format_results([result], ILLUSTRATIVE)


def test_run_check_end_to_end_with_stand_ins(monkeypatch):
    monkeypatch.setenv("TWINCHECK_EXPLAINER", "template")
    monkeypatch.setenv("TWINCHECK_EXTRACTOR", "split")
    ex = EXAMPLES[2]
    out = run_check(ex["message"], "\n".join(ex["evidence"]), ex["lang"], ILLUSTRATIVE)
    assert "confidence" in out and "[1]" in out and ILLUSTRATIVE in out
