"""Tests for src/copies/arabic_twins.py with a fake translator (no GPU)."""

from src.copies import arabic_twins as at

ORIG = {"id": "multifc_1", "source": "multifc", "split": "test", "language": "en",
        "claim": "Obama visited Paris in 2010.", "image_path": None, "label": "supported", "orig_label": "true",
        "evidence": [{"text": "Obama was in Paris in 2010.", "url": "u", "domain": "d"}]}
B = {**ORIG, "id": "multifc_1_B", "label": "contradicted",
     "evidence": [{"text": "Obama was in Rome in 2010.", "url": "u", "domain": "d"}]}
C = {**ORIG, "id": "multifc_1_C", "label": "unverified",
     "evidence": [{"text": "The bridge opened in 1990.", "url": "u2", "domain": "d2"}]}


def fake(prompts, sample=False):
    return ["نص عربي مترجم" for _ in prompts]


def test_clean_name():
    assert at.clean_name(".Oklahoma") == "Oklahoma"
    assert at.clean_name("Cramer’s") == "Cramer"
    assert at.clean_name('... "The Star') == "The Star"
    assert at.clean_name("15th District") == ""


def test_glossary_only_names_in_text():
    g = {"Obama": "أوباما", "Rome": "روما"}
    lines = at.glossary_for("Obama spoke.", g)
    assert "أوباما" in lines and "روما" not in lines
    assert at.glossary_for("Nothing here.", g) == ""


def test_build_twins_types_labels_languages(tmp_path):
    texts = [t for r in (ORIG, B, C) for t in at.texts_of(r)]
    tr, failed = at.translate_texts(texts, {}, fake, 4, str(tmp_path / "c.jsonl"))
    assert failed == []
    twins, dropped = at.build_twins(["multifc_1"], {"multifc_1": ORIG},
                                    {"multifc_1_B": B, "multifc_1_C": C}, tr, set())
    got = {t["id"]: (t["label"], t["language"]) for t in twins}
    assert got == {"multifc_1_A": ("supported", "ar"), "multifc_1_M": ("supported", "mixed"),
                   "multifc_1_B_A": ("contradicted", "ar"), "multifc_1_C_A": ("unverified", "ar")}
    mixed = next(t for t in twins if t["id"].endswith("_M"))
    assert mixed["evidence"][0]["text"] == ORIG["evidence"][0]["text"]  # English evidence kept
    assert mixed["claim"] == "نص عربي مترجم"


def test_failed_translation_drops_twin(tmp_path):
    bad = lambda prompts, sample=False: ["still English text" for _ in prompts]
    tr, failed = at.translate_texts(at.texts_of(C), {}, bad, 4, str(tmp_path / "c.jsonl"))
    assert set(failed) == set(at.texts_of(C))
    tr.update({t: "نص عربي" for t in at.texts_of(ORIG) + at.texts_of(B)})
    twins, dropped = at.build_twins(["multifc_1"], {"multifc_1": ORIG},
                                    {"multifc_1_B": B, "multifc_1_C": C}, tr, set(failed))
    # the claim is shared, so every twin needing the failed claim is dropped
    assert dropped["C_A"] == 1


def test_cache_resumes(tmp_path):
    calls = []
    def counting(prompts, sample=False):
        calls.append(len(prompts))
        return ["نص عربي" for _ in prompts]
    path = str(tmp_path / "c.jsonl")
    at.translate_texts(["a b c", "d e f"], {}, counting, 4, path)
    at.translate_texts(["a b c", "d e f", "g h i"], {}, counting, 4, path)
    assert calls == [2, 1]
