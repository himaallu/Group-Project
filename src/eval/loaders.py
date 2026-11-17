"""Read and check TwinCheck data files and prediction files.

What it does
    * Loads JSON Lines files in the shared record format (CLAUDE.md §4) and checks
      every record against that format.
    * Loads prediction files (CLAUDE.md §5) and checks them.
    * Splits twin ids into base id + twin suffixes, and joins predictions with gold
      labels into one table that ``src/eval/metrics.py`` works on.

Inputs
    * Data files: ``data/processed/**.jsonl``, ``data/copies/**.jsonl``.
    * Prediction files: ``results/<task-id>/preds_<model>.jsonl`` with
      ``{"id", "pred_label", "probs": {"supported", "contradicted", "unverified"}}``.

Outputs
    * Python lists / a pandas DataFrame (no files are written).

How to run (check a file, e.g. Rathi's 10-row sample)
    python -m src.eval.loaders --check data/processed/en/multifc_train.jsonl

Twin id convention
    Twins add suffixes to the source id: ``multifc_000123_B``. Arabic versions of
    evidence twins stack suffixes (``multifc_000123_B_A``). The *twin type* used by
    the metrics is ``B`` or ``C`` whenever one is present (so Arabic swaps still count
    toward ESR), otherwise the last suffix.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

LABELS = ("supported", "contradicted", "unverified")
LANGUAGES = ("en", "ar", "mixed")
TWIN_SUFFIXES = ("P", "A", "M", "B", "C", "V")
REQUIRED_KEYS = ("id", "source", "split", "language", "claim", "evidence",
                 "image_path", "label", "orig_label")
SPLITS = ("train", "val", "test")
PROB_TOLERANCE = 0.01  # probs must sum to 1 within this tolerance


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON Lines file.

    Args:
        path: path to a ``.jsonl`` file (one JSON object per line; blank lines skipped).

    Returns:
        List of dicts, one per line.

    Raises:
        ValueError: if a line is not valid JSON (the message gives the line number).
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e
    return records


def split_twin_id(record_id: str) -> tuple[str, tuple[str, ...]]:
    """Split an id into its base id and its twin suffixes.

    Args:
        record_id: e.g. ``"multifc_000123"``, ``"multifc_000123_B"``,
            ``"multifc_000123_B_A"``.

    Returns:
        ``(base_id, suffixes)``, e.g. ``("multifc_000123", ("B", "A"))``.
        Originals return an empty tuple.
    """
    parts = record_id.split("_")
    suffixes = []
    # Peel suffixes off the end; stop at the first part that is not a twin suffix.
    while len(parts) > 1 and parts[-1] in TWIN_SUFFIXES:
        suffixes.insert(0, parts.pop())
    return "_".join(parts), tuple(suffixes)


def twin_type(record_id: str) -> str | None:
    """Return the twin type the metrics use for this id, or None for an original.

    ``B`` or ``C`` win if present (an Arabic evidence swap is still a swap);
    otherwise the last suffix (``P``, ``A``, ``M`` or ``V``).
    """
    _, suffixes = split_twin_id(record_id)
    if not suffixes:
        return None
    for evidence_twin in ("B", "C"):
        if evidence_twin in suffixes:
            return evidence_twin
    return suffixes[-1]


def validate_record(rec: dict) -> list[str]:
    """Check one record against the shared record format (CLAUDE.md §4).

    Args:
        rec: one parsed JSON record.

    Returns:
        List of human-readable problems; empty if the record is valid.
    """
    problems = []
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    if missing:
        problems.append(f"missing keys: {missing}")

    rid = rec.get("id", "")
    if not isinstance(rid, str) or not rid:
        problems.append("id must be a non-empty string")
    elif "source" in rec and not rid.startswith(str(rec["source"])):
        problems.append(f"id '{rid}' does not start with source '{rec['source']}'")

    if "label" in rec and rec["label"] not in LABELS:
        problems.append(f"label '{rec['label']}' not in {LABELS}")
    if "language" in rec and rec["language"] not in LANGUAGES:
        problems.append(f"language '{rec['language']}' not in {LANGUAGES}")
    if "split" in rec and rec["split"] not in SPLITS:
        problems.append(f"split '{rec['split']}' not in {SPLITS}")
    if "claim" in rec and (not isinstance(rec["claim"], str) or not rec["claim"].strip()):
        problems.append("claim must be a non-empty string")

    evidence = rec.get("evidence")
    if "evidence" in rec:
        if not isinstance(evidence, list):
            problems.append("evidence must be a list")
        else:
            for i, snip in enumerate(evidence):
                if not isinstance(snip, dict):
                    problems.append(f"evidence[{i}] must be an object")
                    continue
                for key in ("text", "url", "domain"):
                    if key not in snip:
                        problems.append(f"evidence[{i}] missing '{key}'")
                if not str(snip.get("text", "")).strip():
                    problems.append(f"evidence[{i}] has empty text")
            if len(evidence) > 10:
                problems.append(f"{len(evidence)} snippets (cleaning keeps <= 10)")
    return problems


def validate_pred(pred: dict) -> list[str]:
    """Check one prediction record against the predictions format (CLAUDE.md §5).

    Args:
        pred: ``{"id", "pred_label", "probs": {"supported", "contradicted", "unverified"}}``.

    Returns:
        List of problems; empty if valid.
    """
    problems = []
    if not isinstance(pred.get("id"), str):
        problems.append("missing or non-string id")
    if pred.get("pred_label") not in LABELS:
        problems.append(f"pred_label '{pred.get('pred_label')}' not in {LABELS}")
    probs = pred.get("probs")
    if not isinstance(probs, dict) or set(probs) != set(LABELS):
        problems.append(f"probs must have exactly the keys {LABELS}")
    else:
        total = sum(float(v) for v in probs.values())
        if abs(total - 1.0) > PROB_TOLERANCE:
            problems.append(f"probs sum to {total:.3f}, not 1")
        if any(float(v) < 0 for v in probs.values()):
            problems.append("negative probability")
    return problems


def load_preds(path: str | Path) -> list[dict]:
    """Load and check a predictions file.

    Args:
        path: ``results/<task-id>/preds_<model>.jsonl``.

    Returns:
        List of prediction dicts.

    Raises:
        ValueError: on the first invalid line or a duplicated id.
    """
    preds = load_jsonl(path)
    seen = set()
    for i, p in enumerate(preds, start=1):
        problems = validate_pred(p)
        if problems:
            raise ValueError(f"{path}: prediction {i} ({p.get('id')}): {problems}")
        if p["id"] in seen:
            raise ValueError(f"{path}: duplicate id {p['id']}")
        seen.add(p["id"])
    return preds


def join_gold(preds: list[dict], gold_records: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Join predictions with gold labels into one table for the metrics.

    Args:
        preds: output of :func:`load_preds`.
        gold_records: shared-format records (originals and/or twins) with ``label``.

    Returns:
        ``(df, missing_ids)`` where ``df`` has one row per prediction that has a gold
        record, with columns ``id, base_id, twin_type, language, gold, pred,
        p_supported, p_contradicted, p_unverified``; ``missing_ids`` lists prediction
        ids with no gold record (they are left out of ``df``).
    """
    gold_by_id = {r["id"]: r for r in gold_records}
    rows, missing = [], []
    for p in preds:
        g = gold_by_id.get(p["id"])
        if g is None:
            missing.append(p["id"])
            continue
        base_id, _ = split_twin_id(p["id"])
        rows.append({
            "id": p["id"],
            "base_id": base_id,
            "twin_type": twin_type(p["id"]),
            "language": g.get("language"),
            "gold": g["label"],
            "pred": p["pred_label"],
            **{f"p_{lab}": float(p["probs"][lab]) for lab in LABELS},
        })
    columns = ["id", "base_id", "twin_type", "language", "gold", "pred",
               "p_supported", "p_contradicted", "p_unverified"]
    return pd.DataFrame(rows, columns=columns), missing


def summarise_file(path: str | Path, max_problems: int = 20) -> dict:
    """Check a shared-format file and summarise it (used for hand-off samples).

    Args:
        path: a shared-format ``.jsonl`` file.
        max_problems: how many problems to keep in the summary.

    Returns:
        Dict with ``n_records``, ``n_invalid``, ``problems`` (first few),
        ``duplicate_ids`` and counts per label / split / language / source /
        twin type, plus snippet-count stats.
    """
    records = load_jsonl(path)
    problems, n_invalid = [], 0
    for i, rec in enumerate(records, start=1):
        rec_problems = validate_record(rec)
        if rec_problems:
            n_invalid += 1
            if len(problems) < max_problems:
                problems.append(f"record {i} ({rec.get('id')}): {'; '.join(rec_problems)}")
    ids = Counter(r.get("id") for r in records)
    n_snippets = [len(r.get("evidence") or []) for r in records]
    return {
        "n_records": len(records),
        "n_invalid": n_invalid,
        "problems": problems,
        "duplicate_ids": [i for i, c in ids.items() if c > 1],
        "label": dict(Counter(r.get("label") for r in records)),
        "split": dict(Counter(r.get("split") for r in records)),
        "language": dict(Counter(r.get("language") for r in records)),
        "source": dict(Counter(r.get("source") for r in records)),
        "twin_type": dict(Counter(str(twin_type(str(r.get("id", "")))) for r in records)),
        "snippets_min_mean_max": (
            (min(n_snippets), round(sum(n_snippets) / len(n_snippets), 2), max(n_snippets))
            if n_snippets else None
        ),
        "n_no_snippets": sum(1 for n in n_snippets if n == 0),
    }


def main() -> None:
    """Command-line entry point: ``python -m src.eval.loaders --check FILE [FILE ...]``."""
    parser = argparse.ArgumentParser(description="Check shared-format JSONL files.")
    parser.add_argument("--check", nargs="+", required=True, help="files to check")
    args = parser.parse_args()
    for path in args.check:
        summary = summarise_file(path)
        print(f"\n== {path}")
        for key, value in summary.items():
            if key == "problems":
                continue
            print(f"  {key}: {value}")
        if summary["problems"]:
            print("  problems:")
            for p in summary["problems"]:
                print(f"    - {p}")
        else:
            print("  problems: none")


if __name__ == "__main__":
    main()
