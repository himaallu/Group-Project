"""Verdict-head features: ~10 numbers per claim from its snippets (H3, CLAUDE.md §3).

What it does
    For every record (original or twin) combines, per snippet, the NLI scores
    (entail / contradict / neutral), the reranker relevance and the entities into
    10 features:

        w_entail, w_contradict, w_neutral  relevance-weighted mean NLI probabilities
        max_entail, max_contradict         highest NLI probability over relevant snippets
        share_relevant                     share of snippets with relevance >= 0.5
        match_place, match_date,           share of the claim's places / dates / people
        match_person                       that appear in the evidence (1.0 if none)
        n_snippets                         number of snippets / 10

    "Relevant" uses REL_THRESHOLD; if no snippet is relevant the max features fall
    back to all snippets. A record with no snippets gets zeros for the NLI features
    (the head learns that this means "unverified").

    Entity match (v0) is string matching of the claim's spaCy entities against the
    snippet text (surname-only mentions count for people, the year for dates).
    It is right for English originals, _P, _B and _C twins (the swap changes the
    evidence, so the claim's entity disappears). TODO(H4): Arabic _A/_M twins need
    the multilingual embedding match from CLAUDE.md §3.

Inputs
    records   shared-format records (data/processed/en/*.jsonl, data/copies/en/*.jsonl)
    nli       DataFrame id, snippet_idx, p_entail, p_contradict, p_neutral
              (features/nli_zeroshot/*.parquet now, Rathi's features/nli_*.parquet later)
    relevance DataFrame id, snippet_idx, relevance (features/relevance_en.parquet)
    entities  {id: {"claim_entities": [{"text", "label"}...]}} (features/entities_en.jsonl)

    Twins look up relevance and entities under their own id first, then under the
    base id (``multifc_000123_B`` -> ``multifc_000123``) for twin types whose snippets
    are the source's (P, A, M, B). _C twins carry another claim's snippets, so they
    need their own relevance rows (Rathi's ``rerank.py`` on the twin file); missing
    ones are filled with ``fill_relevance`` and counted.

Outputs
    DataFrame: id, base_id, twin_type, label, split + FEATURES columns.

How to run (10-row check)
    python -m src.verdict.features --data results/R1/samples/multifc_val_10rows.jsonl \
        --nli features/nli_zeroshot/multifc_val_10rows_first10.parquet
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.loaders import load_jsonl, split_twin_id, twin_type

FEATURES = ("w_entail", "w_contradict", "w_neutral", "max_entail", "max_contradict",
            "share_relevant", "match_place", "match_date", "match_person", "n_snippets")
REL_THRESHOLD = 0.5
ENTITY_GROUPS = {"place": ("GPE", "LOC"), "date": ("DATE",), "person": ("PERSON",)}
SAME_SNIPPETS = ("P", "A", "M", "B")   # twin types that keep the source's snippet order
YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")


# --------------------------------------------------------------------------
# Lookups
# --------------------------------------------------------------------------

def load_entities(path: str | Path) -> dict[str, dict]:
    """{id: entity row} from R2's entities_en.jsonl."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def snippet_table(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[int, tuple]]:
    """{id: {snippet_idx: (values of cols)}} for fast per-record lookups."""
    out: dict[str, dict[int, tuple]] = {}
    values = df[cols].to_numpy()
    for rec_id, idx, row in zip(df["id"], df["snippet_idx"], values):
        out.setdefault(rec_id, {})[int(idx)] = tuple(float(v) for v in row)
    return out


def _lookup(table: dict, rec_id: str) -> dict | None:
    """Rows for a record: its own id first, else the base id for same-snippet twins."""
    if rec_id in table:
        return table[rec_id]
    if twin_type(rec_id) in SAME_SNIPPETS:
        return table.get(split_twin_id(rec_id)[0])
    return None


# --------------------------------------------------------------------------
# Entity match
# --------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lower-case, drop "the"/possessive 's and punctuation, squeeze spaces."""
    text = re.sub(r"'s\b|’s\b", "", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"^the\s+", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def entity_found(entity: dict, evidence_norm: str) -> bool:
    """Does a claim entity appear in the (normalised) evidence text?

    People also match on their surname (last word, >= 3 letters); dates also match
    on any year they contain.
    """
    text = _norm(entity["text"])
    if not text:
        return True
    if re.search(rf"\b{re.escape(text)}\b", evidence_norm):
        return True
    if entity["label"] == "PERSON":
        last = text.split()[-1]
        return len(last) >= 3 and re.search(rf"\b{re.escape(last)}\b", evidence_norm) is not None
    if entity["label"] == "DATE":
        years = YEAR.findall(text)
        return bool(years) and all(y in evidence_norm for y in years)
    return False


def entity_matches(claim_entities: list[dict], evidence: list[dict]) -> dict[str, float]:
    """Share of the claim's places / dates / people found in the evidence.

    Returns:
        {"match_place", "match_date", "match_person"}; 1.0 for a group the claim has
        no entity of (nothing that could be contradicted).
    """
    evidence_norm = " " + _norm(" ".join(e.get("text", "") for e in evidence)) + " "
    out = {}
    for group, labels in ENTITY_GROUPS.items():
        ents = [e for e in claim_entities if e["label"] in labels]
        out[f"match_{group}"] = (float(np.mean([entity_found(e, evidence_norm) for e in ents]))
                                 if ents else 1.0)
    return out


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def record_features(rec: dict, nli_rows: dict | None, rel_rows: dict | None,
                    claim_entities: list[dict], fill_relevance: float = 1.0) -> tuple[dict, int]:
    """The 10 features for one record.

    Args:
        rec: shared-format record.
        nli_rows: {snippet_idx: (p_entail, p_contradict, p_neutral)} or None.
        rel_rows: {snippet_idx: (relevance,)} or None.
        claim_entities: spaCy entities of the claim.
        fill_relevance: relevance used for snippets with no score.

    Returns:
        (feature dict, number of snippets whose relevance was filled in).

    Raises:
        KeyError: if a snippet has no NLI scores (score the file first).
    """
    n = len(rec.get("evidence") or [])
    feats = {f: 0.0 for f in FEATURES}
    feats.update(entity_matches(claim_entities, rec.get("evidence") or []))
    feats["n_snippets"] = n / 10.0
    if n == 0:
        return feats, 0
    nli_rows = nli_rows or {}
    missing_nli = [i for i in range(n) if i not in nli_rows]
    if missing_nli:
        raise KeyError(f"{rec['id']}: no NLI scores for snippets {missing_nli}")
    nli = np.array([nli_rows[i] for i in range(n)])            # (n, 3)
    rel_rows = rel_rows or {}
    rel = np.array([rel_rows[i][0] if i in rel_rows else fill_relevance for i in range(n)])
    n_filled = sum(1 for i in range(n) if i not in rel_rows)

    weights = rel / rel.sum() if rel.sum() > 0 else np.full(n, 1.0 / n)
    feats["w_entail"], feats["w_contradict"], feats["w_neutral"] = (weights @ nli).tolist()
    relevant = rel >= REL_THRESHOLD
    pool = nli[relevant] if relevant.any() else nli
    feats["max_entail"] = float(pool[:, 0].max())
    feats["max_contradict"] = float(pool[:, 1].max())
    feats["share_relevant"] = float(relevant.mean())
    return feats, n_filled


def build_features(records: list[dict], nli: pd.DataFrame, relevance: pd.DataFrame | None,
                   entities: dict[str, dict], fill_relevance: float = 1.0) -> pd.DataFrame:
    """Feature table for many records (see module docstring).

    Args:
        records: shared-format records (originals and/or twins).
        nli: per-snippet NLI scores.
        relevance: per-snippet relevance, or None (= every snippet fully relevant,
            the "no reranker" ablation).
        entities: R2 entity rows by id.
        fill_relevance: value for snippets with no relevance score.

    Returns:
        DataFrame with id, base_id, twin_type, label, split and FEATURES; prints how
        many relevance values were filled in.
    """
    nli_tab = snippet_table(nli, ["p_entail", "p_contradict", "p_neutral"])
    rel_tab = snippet_table(relevance, ["relevance"]) if relevance is not None else {}
    rows, filled = [], 0
    for rec in records:
        ent_row = _lookup(entities, rec["id"]) or entities.get(split_twin_id(rec["id"])[0]) or {}
        feats, n_filled = record_features(
            rec, nli_tab.get(rec["id"]),
            _lookup(rel_tab, rec["id"]) if relevance is not None else None,
            ent_row.get("claim_entities", []),
            fill_relevance if relevance is not None else 1.0)
        if relevance is not None:
            filled += n_filled
        rows.append({"id": rec["id"], "base_id": split_twin_id(rec["id"])[0],
                     "twin_type": twin_type(rec["id"]), "label": rec["label"],
                     "split": rec["split"], **feats})
    if filled:
        print(f"note: {filled} snippets had no relevance score; filled with {fill_relevance}")
    return pd.DataFrame(rows, columns=["id", "base_id", "twin_type", "label", "split", *FEATURES])


def main() -> None:
    """CLI: print the feature table for a (small) data file."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--nli", nargs="+", required=True, help="NLI parquet file(s)")
    ap.add_argument("--relevance", default="features/relevance_en.parquet")
    ap.add_argument("--entities", default="features/entities_en.jsonl")
    ap.add_argument("--out", help="optional CSV output")
    args = ap.parse_args()
    records = [r for p in args.data for r in load_jsonl(p)]
    nli = pd.concat([pd.read_parquet(p) for p in args.nli])
    table = build_features(records, nli, pd.read_parquet(args.relevance),
                           load_entities(args.entities))
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(max_rows=20))
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
