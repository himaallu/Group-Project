"""Evidence-swap twins ``_B`` (R3, CLAUDE.md §2 and §4).

What it does
    For a source claim, choose ONE entity that appears in the claim and in at least
    one snippet: a place (GPE/LOC), a person (PERSON) or a year inside a DATE. Then
    change it in every snippet that mentions it:
        place  -> another place from our own data
        person -> another person from our own data
        year   -> the same year shifted by 2-5 years (never into the future)
    The claim, URLs, domains and all other words stay exactly the same, so the
    evidence now contradicts the claim and the correct verdict is ``contradicted``.

    Replacement names come from the train split's entities. A name must appear at
    least 5 times there (so it is a real, common name), look like a name (letters
    only; people need first + last name), and must not already occur anywhere in the
    record (otherwise the swap could still agree with the claim).

Inputs
    A source record, its entity row from R2 (``features/entities_en.jsonl``) and the
    name pools from ``build_pools``.

Outputs
    (twin record, log row) or None when the claim has no swappable entity.

How to run
    Called by ``src/copies/make_twins.py``.
"""

from __future__ import annotations

import copy
import re
import zlib
import random
from collections import Counter

PLACE_LABELS = {"GPE"}   # cities, states, countries; LOC ("Gulf", "Africa") swaps read oddly
PERSON_LABELS = {"PERSON"}
MIN_POOL_COUNT = 5
YEAR_RE = re.compile(r"\b(19[0-9]{2}|20[0-9]{2})\b")
MAX_YEAR = 2025          # shifted years must stay in the past
YEAR_SHIFTS = (2, 3, 4, 5)
NAME_RE = re.compile(r"^[A-Z][A-Za-z.'\- ]{1,39}$")
# Words that may stand before a surname without making it someone else's name.
TITLES = {"President", "Senator", "Sen", "Rep", "Representative", "Gov", "Governor", "Mr",
          "Mrs", "Ms", "Dr", "Secretary", "Judge", "Justice", "Mayor", "Speaker", "Vice",
          "General", "Gen", "Minister", "Chancellor", "King", "Queen", "Prince", "Former",
          "Candidate", "Congressman", "Congresswoman", "Sheriff", "Chief", "Leader"}


def record_rng(record_id: str, seed: int = 42) -> random.Random:
    """A random generator fixed by the record id and seed.

    Each record gets its own stream, so the twin of a record never depends on
    which other records were processed before it. (crc32, not ``hash``, because
    Python's string hash changes between runs.)
    """
    return random.Random(zlib.crc32(f"{seed}:{record_id}".encode("utf-8")))


def kind_of(label: str) -> str | None:
    """Map a spaCy label to our swap kind: "place", "person", "date" or None."""
    if label in PLACE_LABELS:
        return "place"
    if label in PERSON_LABELS:
        return "person"
    if label == "DATE":
        return "date"
    return None


def looks_like_name(text: str, kind: str) -> bool:
    """True if ``text`` is usable as a replacement name for ``kind``."""
    if not NAME_RE.match(text) or text.isupper():  # all caps = abbreviation ("AK", "USA")
        return False
    return len(text.split()) >= 2 if kind == "person" else True


def build_pools(entity_rows: list[dict], min_count: int = MIN_POOL_COUNT) -> dict:
    """Replacement names from entity rows (use train-split rows only).

    Args:
        entity_rows: rows of ``features/entities_en.jsonl``.
        min_count: a name must be seen at least this often.

    Returns:
        {"place": [names...], "person": [names...]}, sorted for reproducibility.
    """
    counts = {"place": Counter(), "person": Counter()}
    for row in entity_rows:
        for ents in [row["claim_entities"], *row["snippet_entities"]]:
            for ent in ents:
                kind = kind_of(ent["label"])
                if kind in counts and looks_like_name(ent["text"].strip(), kind):
                    counts[kind][ent["text"].strip()] += 1
    return {kind: sorted(name for name, c in counter.items() if c >= min_count)
            for kind, counter in counts.items()}


def mention_pattern(text: str) -> re.Pattern:
    """Regex for whole-word, case-sensitive mentions of ``text``."""
    return re.compile(r"(?<!\w)" + re.escape(text) + r"(?!\w)")


def swap_pairs(kind: str, old: str, new: str) -> list[tuple[str, str, bool]]:
    """The replacements that make up one swap, as (old, new, surname_only) in order.

    Places and years are replaced as they are. People are also mentioned by
    surname alone ("Barack Obama" ... "President Obama said"), so a full name is
    replaced first and then its surname (by the new person's surname). A claim that
    names someone by surname only ("Romney") swaps just the surname, so "Mitt
    Romney" in a snippet becomes "Mitt Kushner", not "Mitt Jared Kushner".
    """
    if kind != "person":
        return [(old, new, False)]
    old_parts, new_parts = old.split(), new.split()
    if len(old_parts) == 1:
        return [(old, new_parts[-1], False)]
    pairs = [(old, new, False)]
    if len(old_parts[-1]) >= 3:          # skip initials and short suffixes ("Jr")
        pairs.append((old_parts[-1], new_parts[-1], True))
    return pairs


def other_persons_name(text: str, start: int) -> bool:
    """True if the word before position ``start`` is a capitalised non-title word.

    After the full name is replaced, a surname still preceded by a first name
    ("Hillary Clinton" when the claim is about Bill Clinton) belongs to someone
    else and must not be swapped.
    """
    words = text[:start].split()
    if not words:
        return False
    prev = words[-1].strip(".,'\"")
    return prev[:1].isupper() and prev not in TITLES and not words[-1].endswith((".", ":"))


def apply_swap(text: str, pairs: list[tuple[str, str, bool]]) -> str:
    """Apply the replacements of ``swap_pairs`` to one snippet (whole words only)."""
    for old, new, surname_only in pairs:
        def replace(match, new=new, surname_only=surname_only):
            """The new value, unless a surname match belongs to another person."""
            if surname_only and other_persons_name(match.string, match.start()):
                return match.group(0)        # someone else's surname: keep it
            return new
        text = mention_pattern(old).sub(replace, text)
    return text


def mentions(kind: str, old: str) -> list[str]:
    """Texts that count as a mention of the entity in a snippet."""
    return [o for o, _, _ in swap_pairs(kind, old, "X Y")]


def candidates(record: dict, entity_row: dict) -> list[tuple[str, str]]:
    """Entities of the claim that can be swapped, as (kind, old_text) pairs.

    Places and people must be mentioned word-for-word (people: full name or
    surname) in at least one snippet. For
    dates, the swappable part is a 4-digit year that also appears in a snippet.
    """
    snippets = [ev["text"] for ev in record["evidence"]]
    found = []
    for ent in entity_row["claim_entities"]:
        kind = kind_of(ent["label"])
        if kind in ("place", "person"):
            old = ent["text"].strip()
            if old and any(mention_pattern(m).search(s)
                           for m in mentions(kind, old) for s in snippets):
                found.append((kind, old))
        elif kind == "date":
            for year in YEAR_RE.findall(ent["text"]):
                if any(mention_pattern(year).search(s) for s in snippets):
                    found.append(("date", year))
    return sorted(set(found))  # unique and in a fixed order before the random choice


def shifted_year(year: str, rng: random.Random) -> str:
    """Year moved by 2-5 years, backwards if forwards would pass ``MAX_YEAR``."""
    shift = rng.choice(YEAR_SHIFTS)
    value = int(year) + shift
    if value > MAX_YEAR:
        value = int(year) - shift
    return str(value)


def pick_replacement(kind: str, old: str, record: dict, pools: dict,
                     rng: random.Random) -> str | None:
    """A new value for the entity that does not occur anywhere in the record."""
    if kind == "date":
        return shifted_year(old, rng)
    full_text = " ".join([record["claim"]] + [ev["text"] for ev in record["evidence"]]).lower()
    old_words = {w.lower() for w in old.split() if len(w) >= 3}
    options = [name for name in pools.get(kind, [])
               if name.lower() not in full_text
               and name.split()[-1].lower() not in full_text   # new surname must be new too
               and not old_words & {w.lower() for w in name.split()}]
    return rng.choice(options) if options else None


def make_swap_twin(record: dict, entity_row: dict, pools: dict,
                   seed: int = 42) -> tuple[dict, dict] | None:
    """Build the ``_B`` twin of one record.

    Args:
        record: source record (shared format).
        entity_row: its row from ``features/entities_en.jsonl``.
        pools: output of ``build_pools``.
        seed: random seed.

    Returns:
        (twin record, log row {"id", "type", "split", "kind", "old", "new",
        "n_snippets_changed"}), or None if nothing can be swapped.
    """
    rng = record_rng(record["id"], seed)
    options = candidates(record, entity_row)
    rng.shuffle(options)
    for kind, old in options:
        new = pick_replacement(kind, old, record, pools, rng)
        if not new:
            continue
        pairs = swap_pairs(kind, old, new)
        twin = copy.deepcopy(record)
        changed = 0
        for ev in twin["evidence"]:
            new_text = apply_swap(ev["text"], pairs)
            if new_text != ev["text"]:
                ev["text"] = new_text
                changed += 1
        if changed == 0:
            continue  # every mention was someone else's surname: try the next entity
        twin["id"] = f"{record['id']}_B"
        twin["label"] = "contradicted"
        log = {"id": twin["id"], "type": "B", "split": record["split"], "kind": kind,
               "old": old, "new": new, "n_snippets_changed": changed}
        return twin, log
    return None
