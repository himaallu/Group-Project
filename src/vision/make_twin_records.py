"""Build shared-format ``_V`` image-twin records from the F2 VERITE and M4FC twins.

What it does
    * Reads the kept F2 VAE twins: VERITE (faces screened, SigLIP 2 cosine >= 0.92) and
      M4FC (faces screened, CLIP and SigLIP 2 cosine >= 0.92).
    * For every twin image, finds **every** source record that uses the original image
      (a VERITE photo has a true and a miscaptioned caption; an M4FC photo has an
      Arabic true caption and a false claim) and copies that record with
      ``id = source id + "_V"`` and ``image_path`` = the twin image.
      Claim, label, evidence, language and split are unchanged, so the authenticity
      leakage metric (ALR, ``src/eval/metrics.py``) compares each twin with its source.
    * Keeps ``method``, ``similarity`` (SigLIP 2 cosine) and ``orig_image_path`` as extra fields.

Inputs
    * ``data/copies/vision/manifests/vae_manifest.csv``: ``image_id, dataset, split,
      method, twin_image_path, similarity``. The image files stay out of git.
    * ``data/copies/vision/m4fc_vae_twins.csv``: M4FC twins with ``both_pass_0_92``.
    * ``data/processed/vision/verite.jsonl`` (``src/vision/load_verite.py``) and
      ``data/processed/ar/m4fc_ar_test.jsonl`` (Y1).

Outputs
    * ``data/copies/vision/vae.jsonl`` (VERITE) and ``data/copies/vision/m4fc_vae.jsonl`` (M4FC).

How to run
    python -m src.vision.make_twin_records
    python -m src.vision.make_twin_records --limit 10      # 10-row test, prints only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

VISION_COPIES = Path("data/copies/vision")
SOURCE_FILES = {
    "VERITE": "data/processed/vision/verite.jsonl",
    "M4FC": "data/processed/ar/m4fc_ar_test.jsonl",
}
SPLIT_NAMES = {"validation": "val"}  # F2 manifests say "validation"; the format says "val"


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON Lines file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def records_by_image(records: list[dict]) -> dict[str, list[dict]]:
    """Group source records by image id (file name without extension).

    Args:
        records: shared-format records with ``image_path``.

    Returns:
        Mapping image id (e.g. ``true_0``, ``m4fc_000002``) -> records using that image.
    """
    by_image: dict[str, list[dict]] = {}
    for rec in records:
        if rec.get("image_path"):
            by_image.setdefault(Path(rec["image_path"]).stem, []).append(rec)
    return by_image


def m4fc_manifest(path: str | Path = VISION_COPIES / "m4fc_vae_twins.csv") -> pd.DataFrame:
    """Kept M4FC twins in the manifest layout used by :func:`twin_records`.

    ``split`` is empty: M4FC twins take the split of their Y1 source records.
    """
    twins = pd.read_csv(path)
    twins = twins[twins["both_pass_0_92"]]
    return pd.DataFrame({"image_id": twins["id"], "dataset": "M4FC", "split": None, "method": "VAE",
                         "twin_image_path": twins["twin_path"], "similarity": twins["siglip2"]})


def twin_records(manifest: pd.DataFrame, by_image: dict[str, list[dict]]) -> list[dict]:
    """Make one ``_V`` record per (twin image, source record using its original).

    Args:
        manifest: twin manifest rows (see module docstring). An empty ``split`` means
            the twin takes the split of its source records.
        by_image: output of :func:`records_by_image` over all source files.

    Returns:
        Shared-format twin records.

    Raises:
        ValueError: if a twin has no source record, or its split disagrees with the
            source's split (twins must keep the split of their source).
    """
    out = []
    for row in manifest.itertuples(index=False):
        sources = by_image.get(row.image_id)
        if not sources:
            raise ValueError(f"no source record for twin image {row.image_id}")
        manifest_split = SPLIT_NAMES.get(row.split, row.split) if pd.notna(row.split) else None
        for src in sources:
            if manifest_split is not None and src["split"] != manifest_split:
                raise ValueError(f"{src['id']}: source split {src['split']} "
                                 f"!= manifest split {manifest_split}")
            twin = dict(src)
            twin["id"] = f"{src['id']}_V"
            twin["image_path"] = row.twin_image_path
            twin["orig_image_path"] = src["image_path"]
            twin["method"] = row.method
            twin["similarity"] = round(float(row.similarity), 4)
            out.append(twin)
    return out


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write records as JSON Lines (UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    """Command-line entry point: build vae.jsonl and m4fc_vae.jsonl."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="only use the first N manifest rows and print instead of writing")
    args = ap.parse_args()

    by_image = {}
    for path in SOURCE_FILES.values():
        by_image.update(records_by_image(load_jsonl(path)))

    manifests = {"vae.jsonl": pd.read_csv(VISION_COPIES / "manifests" / "vae_manifest.csv"),
                 "m4fc_vae.jsonl": m4fc_manifest()}
    for name, manifest in manifests.items():
        if args.limit:
            manifest = manifest.head(args.limit)
        twins = twin_records(manifest, by_image)
        if args.limit:
            for t in twins[:3]:
                print(json.dumps(t, ensure_ascii=False)[:300])
            continue
        write_jsonl(twins, VISION_COPIES / name)
        counts = pd.DataFrame(twins).groupby(["source", "split", "label"]).size()
        print(f"{name}: {len(manifest)} twin images -> {len(twins)} records\n{counts.to_string()}\n")


if __name__ == "__main__":
    main()
