"""Unit tests for the R10 demo polish (no Gradio, GPU or model download needed).

Run: python -m pytest tests/demo/test_r10.py -q
"""

from src.demo.app import claim_card, link_citations, text_dir, verdict_method
from src.explain.generate import build_messages, clean_note


def test_text_direction():
    assert text_dir("Dubai Metro reopened") == "ltr"
    assert text_dir("افتتحت هيئة الطرق محطة جديدة") == "rtl"
    assert text_dir("مترو دبي أوقف all lines بسبب flooding") == "rtl"   # starts in Arabic
    assert text_dir("RTA: مترو دبي يعمل") == "ltr"                    # starts in English
    assert text_dir("") == "ltr"


def test_citations_link_to_sources():
    evidence = [{"text": "a", "url": "https://rta.ae/x", "domain": "rta.ae"},
                {"text": "b", "url": None}]
    out = link_citations("Trains ran <normally> [1]. Rain [2] [7].", evidence)
    assert '<a href="https://rta.ae/x"' in out and "<b>[1]</b></a>" in out
    assert out.count("<a ") == 1 and "<b>[2]</b>" in out   # no URL -> bold, not a link
    assert "[7]" in out and "&lt;normally&gt;" in out      # out of range kept; text escaped


def test_arabic_card_is_right_to_left():
    result = {"claim": "افتتحت هيئة الطرق محطة جديدة", "label": "supported", "confidence": 0.9,
              "explanation": "أعلنت الهيئة افتتاح المحطة [1].",
              "evidence": [{"text": "The RTA opened a station.", "url": "https://a.ae",
                            "domain": "a.ae"}]}
    card = claim_card(result)
    assert card.count('dir="rtl"') == 2                    # claim + note
    assert 'dir="ltr"' in card                             # English snippet stays left-to-right
    assert 'role="progressbar"' in card and 'aria-valuenow="90"' in card
    assert '<a href="https://a.ae" target="_blank" title="a.ae"><b>[1]</b></a>' in card


def test_english_claim_word_removed_from_arabic_notes():
    assert clean_note("الCLAIM خاطئ لأن المترو يعمل [1].", 3) == "الادعاء خاطئ لأن المترو يعمل [1]."
    assert "الادعاء" in clean_note("البيانات لا تؤكد أو تنفي Claim [1].", 3)
    assert clean_note("The claim is wrong [1].", 3) == "The claim is wrong [1]."   # English untouched
    assert "الادعاء" in build_messages("x", "supported", ["a"], "ar")[0]["content"]
    assert "الادعاء" not in build_messages("x", "supported", ["a"], "en")[0]["content"]


def test_verdict_method_names_the_running_step(monkeypatch):
    monkeypatch.setenv("TWINCHECK_VERDICT", "none")       # no model folder -> rule
    assert verdict_method().startswith("evidence rule")


def test_snapshot_renders_saved_examples():
    import json
    from pathlib import Path

    from src.demo.snapshot import cards_html

    report = json.loads(Path("results/R6/examples_rule.json").read_text(encoding="utf-8"))
    page = cards_html(report, ["Metro stopped by floods (mixed)"])
    assert page.count("Illustrative example") == 1
    assert "الادعاء" in page and "CLAIM" not in page          # R10 note fix applied
    assert "Verdict by: evidence rule" in page                 # R6 rule run (verdict stub)
