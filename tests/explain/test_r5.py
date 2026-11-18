"""Unit tests for R5 (cited notes, faithfulness, offline index) on 10 toy rows.

No GPU or model download: Qwen, the NLI model, BGE-M3 and the reranker are faked.
Run: python -m pytest tests/explain -q
"""

import numpy as np
import pytest

from src.explain.faithfulness import judge_note, pick_examples, sentence_citations, summarise
from src.explain.generate import (build_messages, clean_note, generate, split_sentences,
                                  template_note)
from src.retrieval.build_index import collect_snippets, tokenize

SNIPPETS = [{"text": "The Green Line ran normally on Monday."},
            {"text": "Heavy rain fell in Dubai."},
            {"text": "The RTA posted a timetable."}]


def test_clean_note_drops_bad_citations_and_extra_sentences():
    raw = "<think>x</think>Note: The Green Line ran [1]. Rain fell [5]. Third one [2]. Fourth."
    assert clean_note(raw, 3) == "The Green Line ran [1]. Rain fell."


def test_split_sentences_keeps_citation_with_its_sentence():
    assert split_sentences("Trains ran normally. [1] Rain fell in Dubai [2].") == \
        ["Trains ran normally. [1]", "Rain fell in Dubai [2]."]
    assert split_sentences("القطارات تعمل [1]. هطلت أمطار [2]؟") == \
        ["القطارات تعمل [1].", "هطلت أمطار [2]؟"]


def test_prompt_has_numbered_evidence_and_language():
    msgs = build_messages("Metro closed", "contradicted", SNIPPETS, "mixed")
    assert "Modern Standard Arabic" in msgs[0]["content"]
    assert "[3] The RTA posted a timetable." in msgs[1]["content"]
    assert "خاطئ" in msgs[1]["content"]
    assert "English" in build_messages("x", "supported", SNIPPETS, "en")[0]["content"]


def test_template_backend_without_gpu(monkeypatch):
    monkeypatch.setenv("TWINCHECK_EXPLAINER", "template")
    note = generate("Metro closed", "contradicted", SNIPPETS + [{"text": "4th"}], "en")
    assert note.startswith("Verdict: contradicted") and "[1]" in note
    assert "[1]" in template_note("supported", ["نص"], "ar")
    assert "No relevant evidence" in template_note("unverified", [], "en")


def fake_nli(claims, snippets):
    """Entailed if every word of the claim (hypothesis) is in the snippet."""
    return [{"p_entail": 0.9 if set(c.lower().strip(".").split()) <= set(s.lower().strip(".").split())
             else 0.1, "p_contradict": 0.05, "p_neutral": 0.05} for c, s in zip(claims, snippets)]


def test_sentence_citations():
    assert sentence_citations("Rain fell in Dubai [2][2] [1].") == ("Rain fell in Dubai.", [2, 1])


def test_judge_note_faithfulness_and_citations():
    note = "Heavy rain fell in Dubai [2]. The RTA cancelled trains [3]. No citation here."
    result = judge_note(note, SNIPPETS, fake_nli)
    assert result["sentence_entailed"] == [True, False, False]
    assert result["citation_supports"] == [True, False]
    # a sentence citing a supporting and a non-supporting snippet is faithful
    two = judge_note("Heavy rain fell in Dubai [1][2].", SNIPPETS, fake_nli)
    assert two["sentence_entailed"] == [True] and two["citation_supports"] == [False, True]


def test_summarise_and_examples():
    rows = []
    for i in range(10):
        ok = i % 2 == 0
        rows.append({"id": f"r{i}", "language": "en" if i < 8 else "ar",
                     "sentence_entailed": [True, ok], "citation_supports": [True, ok]})
    rep = summarise(rows)
    assert rep["all"]["faithfulness"] == 0.75 and rep["all"]["citation_accuracy"] == 0.75
    assert rep["all"]["share_notes_fully_faithful"] == 0.5
    assert rep["ar"]["n_notes"] == 2
    good, bad = pick_examples(rows)
    assert len(good) == 5 and len(bad) == 5
    assert all(all(r["sentence_entailed"]) for r in good)


def test_collect_snippets_dedupes_and_keeps_ids():
    recs = [{"id": "a", "language": "en", "evidence": [{"text": "same text", "url": "u"},
                                                       {"text": " "}]},
            {"id": "b", "language": "en", "evidence": [{"text": "same text "}, {"text": "other"}]}]
    snips = collect_snippets(recs)
    assert [s["text"] for s in snips] == ["same text", "other"]
    assert snips[0]["record_ids"] == ["a", "b"] and snips[0]["url"] == "u"
    assert tokenize("Dubai Metro, مترو دبي!") == ["dubai", "metro", "مترو", "دبي"]


def test_offline_index_hybrid_search():
    pytest.importorskip("rank_bm25")
    pytest.importorskip("faiss")
    from src.retrieval.build_index import OfflineIndex, recall_at_k

    recs = [{"id": f"r{i}", "language": "en", "claim": f"topic{i} happened",
             "evidence": [{"text": f"report about topic{i} number {j}"} for j in range(2)]}
            for i in range(10)]
    snips = collect_snippets(recs)
    vocab = [f"topic{i}" for i in range(10)]

    def encode(texts):
        vecs = np.array([[1.0 if w in t else 0.0 for w in vocab] for t in texts]) + 1e-3
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def rerank(claim, texts):
        return [1.0 if claim.split()[0] in t else 0.0 for t in texts]

    index = OfflineIndex(snips, encode([s["text"] for s in snips]), encode, rerank)
    hits = index.search("topic3 happened", k=2)
    assert all("r3" in h["record_ids"] for h in hits) and hits[0]["score"] == 1.0
    assert recall_at_k(index, recs, k=2) == 1.0
    assert len(index.search("topic3", k=3, mode="bm25")) == 3
