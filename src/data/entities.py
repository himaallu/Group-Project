"""Named entities (who / where / when) for claims and snippets with spaCy (R2).

What it does
    Runs spaCy ``en_core_web_trf`` on every claim and every snippet and keeps the
    entity types we use: PERSON, GPE, LOC, ORG, DATE, EVENT. Character offsets are
    kept so R3 can swap an entity inside a snippet without touching anything else.
    Used by R3 (evidence-swap twins, "no shared entities" for unrelated twins,
    paraphrase checks) and the verdict-head entity-match features (H3).

Inputs
    JSONL files in the shared record format (data/processed/en/*.jsonl).

Outputs
    One JSONL file, one line per record:
    {"id": "multifc_000123",
     "claim_entities": [{"text": "Dubai", "label": "GPE", "start": 0, "end": 5}],
     "snippet_entities": [[...], [...]]}   # one list per evidence snippet, same order

How to run (Colab GPU; ~30 min for all English files on a T4)
    python -m src.data.entities data/processed/en/multifc_val.jsonl --limit 10 \
        --out features/test10/entities_en.jsonl
    python -m src.data.entities data/processed/en/{multifc,averitec}_{train,val,test}.jsonl \
        --out features/entities_en.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.eval.loaders import load_jsonl

KEEP_LABELS = ("PERSON", "GPE", "LOC", "ORG", "DATE", "EVENT")
DEFAULT_MODEL = "en_core_web_trf"


def doc_entities(doc) -> list[dict]:
    """Entities of one spaCy Doc, limited to ``KEEP_LABELS``.

    Args:
        doc: a spaCy Doc (anything with ``.ents`` whose items have ``text``,
            ``label_``, ``start_char``, ``end_char``).

    Returns:
        [{"text", "label", "start", "end"}, ...] in text order.
    """
    return [{"text": ent.text, "label": ent.label_, "start": ent.start_char,
             "end": ent.end_char} for ent in doc.ents if ent.label_ in KEEP_LABELS]


def load_nlp(model: str = DEFAULT_MODEL):
    """Load spaCy on the GPU if there is one, keeping only the NER component."""
    import spacy

    spacy.prefer_gpu()
    nlp = spacy.load(model)
    # Only NER is needed; switching the rest off makes it about 2x faster.
    keep = {"transformer", "tok2vec", "ner"}
    for name in list(nlp.pipe_names):
        if name not in keep:
            nlp.disable_pipe(name)
    return nlp


def record_entities(records: list[dict], nlp, batch_size: int = 128) -> list[dict]:
    """Entities for the claim and every snippet of each record.

    All texts go through ``nlp.pipe`` in one stream (fast batching), then are put
    back per record.

    Args:
        records: shared-format records.
        nlp: a loaded spaCy pipeline (or any callable with ``pipe``).
        batch_size: texts per spaCy batch.

    Returns:
        One output dict per record (format in the module docstring).
    """
    texts, owners = [], []  # owners[i] = (record index, snippet index or None for the claim)
    for r, rec in enumerate(records):
        texts.append(rec["claim"])
        owners.append((r, None))
        for s, ev in enumerate(rec.get("evidence") or []):
            texts.append(ev.get("text") or "")
            owners.append((r, s))

    out = [{"id": rec["id"], "claim_entities": [],
            "snippet_entities": [[] for _ in rec.get("evidence") or []]} for rec in records]
    for (r, s), doc in zip(owners, nlp.pipe(texts, batch_size=batch_size)):
        if s is None:
            out[r]["claim_entities"] = doc_entities(doc)
        else:
            out[r]["snippet_entities"][s] = doc_entities(doc)
    return out


def main() -> None:
    """CLI: extract entities from files and write one JSONL file."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="shared-format JSONL files")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="features/entities_en.jsonl")
    ap.add_argument("--limit", type=int, help="first N records per file (10-row check)")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    nlp = load_nlp(args.model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for path in args.files:
            records = load_jsonl(path)
            records = records[:args.limit] if args.limit else records
            start = time.time()
            for row in record_entities(records, nlp, args.batch_size):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(records)
            print(f"{path}: {len(records)} records ({time.time() - start:.0f}s)")
    print(f"{total} records -> {args.out}")


if __name__ == "__main__":
    main()
