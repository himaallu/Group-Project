"""Unrelated-evidence twins ``_C`` (R3, CLAUDE.md §2 and §4).

What it does
    Replaces a source claim's evidence with the snippets of a *donor* claim from the
    same split. A donor is accepted only if
        1. it shares no entity with the source claim (entity texts compared in lower
           case over the donor's claim and all its snippets), and
        2. every donor snippet scores below 0.1 with the reranker against the source
           claim.
    Up to 20 random donors are tried per source (fixed by seed). The claim stays the
    same, so with this evidence the correct verdict is ``unverified``.

Inputs
    Source record, the pool of donor records of the same split, the R2 entity rows,
    and a scoring function ``score(claim, snippets) -> list[float]``
    (``src.retrieval.rerank.Reranker.score`` on Colab; a fake in the tests).

Outputs
    (twin record, log row) or None if no donor passed.

How to run
    Called by ``src/copies/make_twins.py``.
"""

from __future__ import annotations

import copy

from src.copies.swap import record_rng

MAX_DONOR_RELEVANCE = 0.1
MAX_TRIES = 20


def entity_texts(entity_row: dict, claim_only: bool = False) -> set[str]:
    """Lower-cased entity texts of a record (claim, plus snippets unless claim_only)."""
    groups = [entity_row["claim_entities"]]
    if not claim_only:
        groups += entity_row["snippet_entities"]
    return {ent["text"].strip().lower() for ents in groups for ent in ents if ent["text"].strip()}


def make_unrelated_twin(record: dict, donors: list[dict], entities: dict, score,
                        seed: int = 42, max_tries: int = MAX_TRIES
                        ) -> tuple[dict, dict] | None:
    """Build the ``_C`` twin of one record.

    Args:
        record: source record.
        donors: candidate donor records (same split; must have evidence).
        entities: {id: entity row} from ``features/entities_en.jsonl``.
        score: function (claim, list of snippets) -> list of relevance scores.
        seed: random seed.
        max_tries: donors to try before giving up.

    Returns:
        (twin record, log row {"id", "type", "split", "donor_id", "max_relevance",
        "tries"}), or None.
    """
    rng = record_rng(record["id"], seed + 1)  # +1: a different stream than the _B twin
    source_names = entity_texts(entities[record["id"]], claim_only=True)
    claim_key = record["claim"].strip().lower()
    tries = 0
    for _ in range(max_tries * 5):  # skips for shared entities do not use up a try
        if tries >= max_tries:
            break
        donor = donors[rng.randrange(len(donors))]
        if donor["id"] == record["id"] or donor["claim"].strip().lower() == claim_key \
                or not donor["evidence"]:
            continue
        if source_names & entity_texts(entities[donor["id"]]):
            continue
        tries += 1
        scores = score(record["claim"], [ev["text"] for ev in donor["evidence"]])
        if max(scores) < MAX_DONOR_RELEVANCE:
            twin = copy.deepcopy(record)
            twin["id"] = f"{record['id']}_C"
            twin["evidence"] = copy.deepcopy(donor["evidence"])
            twin["label"] = "unverified"
            log = {"id": twin["id"], "type": "C", "split": record["split"],
                   "donor_id": donor["id"], "max_relevance": round(max(scores), 4),
                   "tries": tries}
            return twin, log
    return None
