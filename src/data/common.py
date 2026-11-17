"""Shared helpers for the TwinCheck data scripts (R1).

What it does
    Small, reusable functions that every loader and ``clean.py`` use:
    reading / writing JSON Lines, building a record in the shared format
    (CLAUDE.md section 4), reading ``data/label_map.csv``, getting the domain of a
    URL, and a stratified train/val/test split with a fixed seed.

Inputs / outputs
    No files of its own; imported by ``load_multifc.py``, ``load_averitec.py`` and
    ``clean.py``.

How to run
    Not run directly. Example:
        from src.data.common import read_jsonl, make_record
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from urllib.parse import urlparse

LABELS = ("supported", "contradicted", "unverified")
SPLITS = ("train", "val", "test")
DROPPED = "dropped"  # label-map value for claims we leave out (half true, satire, ...)
SEED = 42

# Default location of the label map, relative to the repo root.
LABEL_MAP_PATH = os.path.join("data", "label_map.csv")


def read_jsonl(path: str) -> list[dict]:
    """Read a JSON Lines file.

    Args:
        path: path to a ``.jsonl`` file (one JSON object per line).

    Returns:
        List of dicts, one per non-empty line.
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: str) -> None:
    """Write records to a JSON Lines file, creating the folder if needed.

    Args:
        records: list of dicts.
        path: output ``.jsonl`` path. Arabic text is written as-is (not escaped).
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def domain_of(url: str) -> str:
    """Return the lower-case domain of a URL without a leading ``www.``.

    Args:
        url: full URL, e.g. ``"https://www.gulfnews.com/uae/..."``.

    Returns:
        Domain such as ``"gulfnews.com"``; empty string if the URL has none.
    """
    if not url:
        return ""
    netloc = urlparse(url if "//" in url else "//" + url).netloc.lower()
    netloc = netloc.split("@")[-1].split(":")[0]  # drop user info and port
    if netloc == "web.archive.org":
        # Archived page: .../web/<timestamp>/<original url>. Use the original site.
        inner = url.split("/web/", 1)[-1].split("/", 1)
        if len(inner) == 2 and inner[1]:
            return domain_of(inner[1])
    return netloc[4:] if netloc.startswith("www.") else netloc


def make_evidence(text: str, url: str) -> dict:
    """Build one evidence snippet ``{"text", "url", "domain"}``.

    Args:
        text: snippet text.
        url: source URL (may be empty).

    Returns:
        Evidence dict in the shared format.
    """
    url = (url or "").strip()
    return {"text": " ".join(text.split()), "url": url, "domain": domain_of(url)}


def make_record(rec_id: str, source: str, split: str, claim: str,
                evidence: list[dict], label: str, orig_label: str,
                language: str = "en", image_path: str | None = None) -> dict:
    """Build one record in the shared format (CLAUDE.md section 4).

    Args:
        rec_id: id starting with the source name, e.g. ``"multifc_000123"``.
        source: dataset name, e.g. ``"multifc"``.
        split: ``"train"``, ``"val"`` or ``"test"``.
        claim: claim text (whitespace is normalised).
        evidence: list of evidence dicts from ``make_evidence``.
        label: ``supported`` / ``contradicted`` / ``unverified``.
        orig_label: the dataset's own label, kept for reference.
        language: ``"en"``, ``"ar"`` or ``"mixed"``.
        image_path: path to an image, or None.

    Returns:
        Record dict with keys in the shared-format order.
    """
    return {
        "id": rec_id,
        "source": source,
        "split": split,
        "language": language,
        "claim": " ".join(claim.split()),
        "evidence": evidence,
        "image_path": image_path,
        "label": label,
        "orig_label": orig_label,
    }


def normalise_label(label: str) -> str:
    """Lower-case and trim a dataset label so map lookups are robust."""
    return " ".join((label or "").strip().lower().split())


def load_label_map(path: str = LABEL_MAP_PATH) -> dict[tuple[str, str], str]:
    """Read ``data/label_map.csv`` (columns ``source, orig_label, label``).

    Lines starting with ``#`` are comments.

    Args:
        path: path to the CSV file.

    Returns:
        Dict ``(source, normalised orig_label) -> label`` where label is one of
        ``LABELS`` or ``"dropped"``.
    """
    mapping = {}
    with open(path, encoding="utf-8", newline="") as f:
        rows = (line for line in f if line.strip() and not line.startswith("#"))
        for row in csv.DictReader(rows):
            label = row["label"].strip()
            if label not in LABELS + (DROPPED,):
                raise ValueError(f"{path}: bad label {label!r} for {row}")
            mapping[(row["source"].strip(), normalise_label(row["orig_label"]))] = label
    return mapping


def map_label(label_map: dict[tuple[str, str], str], source: str,
              orig_label: str) -> str | None:
    """Map a dataset label to our 3 labels.

    Args:
        label_map: output of ``load_label_map``.
        source: dataset name used in the CSV, e.g. ``"multifc"``.
        orig_label: the dataset's label.

    Returns:
        ``supported`` / ``contradicted`` / ``unverified``, ``"dropped"``, or None
        when the label is not in the map (callers count and drop these).
    """
    return label_map.get((source, normalise_label(orig_label)))


def stratified_split(records: list[dict], ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
                     names: tuple[str, ...] = SPLITS, seed: int = SEED,
                     key: str = "label") -> dict[str, list[dict]]:
    """Split records into parts with the same label mix in each part.

    Plain-Python version of a stratified split (no scikit-learn needed). Within
    each label the records are shuffled with ``seed`` and cut by ``ratios``.

    Args:
        records: list of dicts.
        ratios: share of each part; must sum to 1.
        names: name of each part (same length as ``ratios``).
        seed: random seed (42 for the project).
        key: field to stratify on.

    Returns:
        Dict ``name -> list of records``. Each record's ``split`` field is NOT
        changed here; the caller sets it.
    """
    assert len(ratios) == len(names) and abs(sum(ratios) - 1) < 1e-9
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for rec in records:
        by_label[rec[key]].append(rec)

    parts = {name: [] for name in names}
    for label in sorted(by_label):  # sorted so the result does not depend on dict order
        group = by_label[label][:]
        rng.shuffle(group)
        start = 0
        for i, (name, ratio) in enumerate(zip(names, ratios)):
            # The last part takes the remainder so no record is lost to rounding.
            end = len(group) if i == len(names) - 1 else start + round(ratio * len(group))
            parts[name].extend(group[start:end])
            start = end
    return parts
