"""Unit tests for R2 (reranker pairs, AUROC, scoring order, entity output) on 10 rows.

No GPU or model download: spaCy and the reranker are replaced by small fakes.
Run: python -m pytest tests/retrieval -q
"""

from types import SimpleNamespace

import pytest

from src.data.entities import doc_entities, record_entities
from src.retrieval.build_pairs import binary_auroc, make_pairs
from src.retrieval.rerank import length_order, snippet_rows


def toy_records(n: int = 10) -> list[dict]:
    """n records with 2 snippets each; snippet text names its own claim."""
    return [{"id": f"toy_{i:06d}", "claim": f"claim number {i}",
             "evidence": [{"text": f"snippet {j} of claim {i}", "url": "", "domain": ""}
                          for j in range(2)]} for i in range(n)]


def test_pairs_one_relevant_one_irrelevant_per_claim():
    records = toy_records()
    pairs = make_pairs(records, n_claims=5)
    assert len(pairs) == 10
    assert sum(p["label"] for p in pairs) == 5
    for p in pairs:
        own = p["snippet"].endswith(f"claim {int(p['claim_id'][-6:])}")
        assert own == (p["label"] == 1)          # relevant iff the snippet is the claim's own
        assert (p["snippet_from"] == p["claim_id"]) == (p["label"] == 1)


def test_pairs_reproducible_and_seed_dependent():
    records = toy_records()
    assert make_pairs(records, 5, seed=42) == make_pairs(records, 5, seed=42)
    assert make_pairs(records, 5, seed=42) != make_pairs(records, 5, seed=43)


def test_negative_never_from_duplicate_claim_text():
    records = toy_records(3)
    records[1]["claim"] = records[0]["claim"].upper()   # same claim, different case
    for p in make_pairs(records, seed=1):
        if p["label"] == 0 and p["claim_id"] in ("toy_000000", "toy_000001"):
            assert p["snippet_from"] == "toy_000002"


def test_pairs_need_two_records():
    with pytest.raises(ValueError):
        make_pairs(toy_records(1))


def test_binary_auroc():
    assert binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert binary_auroc([0, 1], [0.5, 0.5]) == 0.5        # tie counts half
    assert binary_auroc([1, 1], [0.1, 0.2]) is None


def test_snippet_rows_and_length_order():
    records = toy_records(2)
    records[0]["evidence"].append({"text": "  ", "url": "", "domain": ""})  # empty: skipped
    rows = snippet_rows(records)
    assert [(r["id"], r["snippet_idx"]) for r in rows] == [
        ("toy_000000", 0), ("toy_000000", 1), ("toy_000001", 0), ("toy_000001", 1)]
    assert length_order(["aaa", "a", "aa"], ["", "", ""]) == [1, 2, 0]


class FakeNLP:
    """Stands in for spaCy: every capitalised word is a GPE entity."""

    def pipe(self, texts, batch_size=128):
        for text in texts:
            ents, pos = [], 0
            for word in text.split(" "):
                if word[:1].isupper():
                    ents.append(SimpleNamespace(text=word, label_="GPE",
                                                start_char=pos, end_char=pos + len(word)))
                pos += len(word) + 1
            ents.append(SimpleNamespace(text="x", label_="CARDINAL", start_char=0, end_char=1))
            yield SimpleNamespace(ents=ents)


def test_entities_offsets_labels_and_alignment():
    records = [{"id": "toy_1", "claim": "Flooding in Dubai today",
                "evidence": [{"text": "Rain hit Sharjah"}, {"text": "no names here"}]}]
    out = record_entities(records, FakeNLP())
    assert out[0]["id"] == "toy_1"
    assert out[0]["claim_entities"] == [
        {"text": "Flooding", "label": "GPE", "start": 0, "end": 8},
        {"text": "Dubai", "label": "GPE", "start": 12, "end": 17}]
    assert [e["text"] for e in out[0]["snippet_entities"][0]] == ["Rain", "Sharjah"]
    assert out[0]["snippet_entities"][1] == []           # CARDINAL is filtered out
    claim = records[0]["claim"]
    for ent in out[0]["claim_entities"]:
        assert claim[ent["start"]:ent["end"]] == ent["text"]


def test_doc_entities_filters_labels():
    doc = SimpleNamespace(ents=[SimpleNamespace(text="2019", label_="DATE", start_char=0,
                                                end_char=4),
                                SimpleNamespace(text="5", label_="CARDINAL", start_char=5,
                                                end_char=6)])
    assert doc_entities(doc) == [{"text": "2019", "label": "DATE", "start": 0, "end": 4}]
