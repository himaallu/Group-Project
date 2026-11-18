"""Unit tests for the R3 twin generators on 10 toy rows (no GPU, no model downloads).

spaCy, the reranker and Qwen are replaced by small fakes.
Run: python -m pytest tests/copies -q
"""

from types import SimpleNamespace

from src.copies.checks import check_twin
from src.copies.paraphrase import clean_output, is_good_paraphrase, make_paraphrase_twins
from src.copies.sources import is_eligible, select_sources
from src.copies.swap import build_pools, candidates, make_swap_twin, shifted_year, record_rng
from src.copies.unrelated import make_unrelated_twin

PLACES = ["Dubai", "Paris", "Texas", "Ohio", "Berlin", "Lagos", "Tokyo", "Cairo", "Oslo", "Lima"]


def ent(text: str, label: str, claim: str) -> dict:
    """Entity dict with offsets into ``claim``."""
    start = claim.find(text)
    return {"text": text, "label": label, "start": start, "end": start + len(text)}


def toy(i: int) -> tuple[dict, dict]:
    """Record i (supported, 2 snippets naming PLACES[i] and year 2010+i) and its entity row."""
    place, year = PLACES[i], str(2010 + i)
    claim = f"Floods closed every school in {place} in {year}"
    record = {"id": f"multifc_{i:06d}", "source": "multifc", "split": "test",
              "language": "en", "claim": claim, "image_path": None,
              "label": "supported", "orig_label": "true",
              "evidence": [{"text": f"Schools in {place} shut after floods in {year}.",
                            "url": "u", "domain": "d"},
                           {"text": f"{place} officials said rain was heavy.",
                            "url": "u", "domain": "d"}]}
    row = {"id": record["id"],
           "claim_entities": [ent(place, "GPE", claim), ent(year, "DATE", claim)],
           "snippet_entities": [[{"text": place, "label": "GPE", "start": 11,
                                  "end": 11 + len(place)}],
                                [{"text": place, "label": "GPE", "start": 0,
                                  "end": len(place)}]]}
    return record, row


RECORDS, ROWS = zip(*(toy(i) for i in range(10)))
ENTITIES = {row["id"]: row for row in ROWS}
POOLS = {"place": PLACES, "person": ["Jane Smith"]}


def test_sources_need_supported_and_two_relevant_snippets():
    rel = {r["id"]: {0: 0.9, 1: 0.8} for r in RECORDS}
    rel["multifc_000001"] = {0: 0.9, 1: 0.2}                 # only 1 relevant snippet
    contradicted = dict(RECORDS[2], label="contradicted")
    assert is_eligible(RECORDS[0], rel)
    assert not is_eligible(RECORDS[1], rel)
    assert not is_eligible(contradicted, rel)
    chosen = select_sources([list(RECORDS[:5]), list(RECORDS[5:])], rel, n=7)
    assert len(chosen) == 7 and "multifc_000001" not in {c["id"] for c in chosen}
    assert {c["id"] for c in chosen} >= {r["id"] for r in RECORDS[:5] if r["id"] != "multifc_000001"}
    assert chosen == select_sources([list(RECORDS[:5]), list(RECORDS[5:])], rel, n=7)


def test_build_pools_min_count_and_names():
    rows = [{"claim_entities": [{"text": "Paris", "label": "GPE"}, {"text": "Gulf", "label": "LOC"},
                                {"text": "Bob", "label": "PERSON"},
                                {"text": "Jane Smith", "label": "PERSON"}],
             "snippet_entities": []}] * 5
    rows.append({"claim_entities": [{"text": "Rare Town", "label": "GPE"}], "snippet_entities": []})
    pools = build_pools(rows)
    assert pools == {"place": ["Paris"], "person": ["Jane Smith"]}  # Bob: one word only


def test_swap_changes_only_the_entity_everywhere():
    record, row = RECORDS[0], ROWS[0]
    assert set(candidates(record, row)) == {("place", "Dubai"), ("date", "2010")}
    twin, log = make_swap_twin(record, row, POOLS)
    assert twin["id"] == "multifc_000000_B" and twin["label"] == "contradicted"
    assert twin["claim"] == record["claim"]
    assert log["n_snippets_changed"] >= 1
    for old_ev, new_ev in zip(record["evidence"], twin["evidence"]):
        assert new_ev["text"] == old_ev["text"].replace(log["old"], log["new"])
        assert new_ev["url"] == old_ev["url"]
    assert log["new"] not in record["claim"]
    assert check_twin(twin, {record["id"]: record}, {twin["id"]: log}, ENTITIES) == []
    assert make_swap_twin(record, row, POOLS) == (twin, log)        # reproducible


def test_year_shift_stays_in_range_and_past():
    for i in range(50):
        new = int(shifted_year("2023", record_rng(str(i))))
        assert 2 <= abs(new - 2023) <= 5 and new <= 2025


def test_no_swap_without_shared_entity():
    record, row = RECORDS[0], dict(ROWS[0], claim_entities=[])
    assert make_swap_twin(record, row, POOLS) is None


def test_unrelated_twin_respects_entities_and_relevance():
    record = RECORDS[0]
    donors = list(RECORDS)

    def low(claim, snippets):
        return [0.05] * len(snippets)

    twin, log = make_unrelated_twin(record, donors, ENTITIES, low)
    assert twin["id"] == "multifc_000000_C" and twin["label"] == "unverified"
    assert log["donor_id"] != record["id"]
    assert twin["evidence"] == next(d for d in donors if d["id"] == log["donor_id"])["evidence"]
    originals = {r["id"]: r for r in RECORDS}
    assert check_twin(twin, originals, {twin["id"]: log}, ENTITIES) == []

    def high(claim, snippets):
        return [0.5] * len(snippets)

    assert make_unrelated_twin(record, donors, ENTITIES, high) is None
    # Donors that mention the source's place are never used.
    sharing = {k: dict(v, snippet_entities=[[{"text": "Dubai", "label": "GPE"}]])
               for k, v in ENTITIES.items()}
    assert make_unrelated_twin(record, donors, sharing, low) is None
    assert make_unrelated_twin(record, [record], ENTITIES, low) is None   # only itself


class FakeNLP:
    """Capitalised words and numbers are entities."""

    def pipe(self, texts):
        for text in texts:
            yield SimpleNamespace(ents=[SimpleNamespace(text=w.strip(".,"), label_="GPE")
                                        for w in text.split() if w[:1].isupper() or w.isdigit()])


def test_paraphrase_checks():
    nlp = FakeNLP()
    claim = "floods closed schools in Dubai in 2019"
    good = "schools were shut by floods in Dubai in 2019"
    assert not is_good_paraphrase(claim, claim + ".", *nlp.pipe([claim, claim + "."]))
    assert not is_good_paraphrase(claim, "In Paris floods closed schools in 2019",
                                  *nlp.pipe([claim, "In Paris floods closed schools in 2019"]))
    ok = "floods closed the schools of Dubai in 2019"
    assert is_good_paraphrase(claim, ok, *nlp.pipe([claim, ok]))
    assert not is_good_paraphrase(claim, good.replace("2019", "2018"), *nlp.pipe([claim, good]))
    assert clean_output('Reworded claim: "Hello world."\nextra') == "Hello world."


def test_paraphrase_retry_then_drop():
    nlp = FakeNLP()
    recs = [dict(RECORDS[0], claim="rain hit Dubai in 2019", id="multifc_000000"),
            dict(RECORDS[1], claim="rain hit Paris in 2019", id="multifc_000001")]
    calls = []

    def generate(claims, sample):
        calls.append(sample)
        # first claim: good on the first try; second: wrong place every time
        return ["in 2019 rain struck Dubai" if "Dubai" in c else "rain hit Oslo in 2019"
                for c in claims]

    pairs, stats = make_paraphrase_twins(recs, generate, nlp)
    assert [t["id"] for t, _ in pairs] == ["multifc_000000_P"]
    assert pairs[0][0]["evidence"] == recs[0]["evidence"]
    assert stats["accepted_attempt_1"] == 1 and stats["dropped_failed_checks"] == 1
    assert calls == [False, True, True]


def test_person_swap_also_replaces_surname_only_mentions():
    from src.copies.swap import apply_swap, swap_pairs
    pairs = swap_pairs("person", "Barack Obama", "Jane Smith")
    text = "Barack Obama spoke. Later President Obama left. Obamacare stayed. Michelle Obama too."
    assert apply_swap(text, pairs) ==         "Jane Smith spoke. Later President Smith left. Obamacare stayed. Michelle Obama too."
    # surname-only claim: only the surname is swapped
    assert apply_swap("Mitt Romney won", swap_pairs("person", "Romney", "Jane Smith")) == \
        "Mitt Smith won"


def test_pools_skip_abbreviations():
    rows = [{"claim_entities": [{"text": "AK", "label": "GPE"}, {"text": "Alaska", "label": "GPE"}],
             "snippet_entities": []}] * 5
    assert build_pools(rows)["place"] == ["Alaska"]
