"""Build shared-format ``_V`` image-twin records from the F2 twin manifests.

What it does
    * Reads a twin manifest (one row per twin image made in F2: VAE round-trip or
      SDXL img2img, already filtered by SigLIP cosine >= 0.92 and face screening).
    * For every twin image, finds **every** source record that uses the original image
      (a VERITE photo appears with its true caption and a miscaptioned one; an M4FC
      photo can carry several claims) and copies that record with
      ``id = source id + "_V"`` and ``image_path`` = the twin image.
      Claim, label, evidence, language and split are unchanged, so the authenticity
      leakage metric (ALR, ``src/eval/metrics.py``) compares each twin with its source.
    * Keeps ``method`` (VAE / SDXL), ``similarity`` and ``orig_image_path`` as extra fields.
    * Rewrites ``data/copies/vision/sources.csv`` (SDXL source images) with repo-relative
      paths and shared-format labels and splits.

Inputs
    * ``data/copies/vision/manifests/{vae,sdxl}_manifest.csv``: ``image_id, dataset,
      split, method, twin_image_path, jpeg85_original_path, similarity`` (kept twins),
      and ``manifests/sdxl_sources_manifest.csv`` (all approved SDXL sources).
      These are the F2 notebook outputs; the image files stay out of git (public repo).
    * ``data/processed/vision/verite.jsonl`` (``src/vision/load_verite.py``) and
      ``data/processed/ar/m4fc_ar_test.jsonl`` (Youssef, Y1).

Outputs
    * ``data/copies/vision/vae.jsonl`` and ``data/copies/vision/sdxl.jsonl``.
    * ``data/copies/vision/sources.csv``.

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


def twin_records(manifest: pd.DataFrame, by_image: dict[str, list[dict]]) -> list[dict]:
    """Make one ``_V`` record per (twin image, source record using its original).

    Args:
        manifest: twin manifest rows (see module docstring).
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
        manifest_split = SPLIT_NAMES.get(row.split, row.split)
        for src in sources:
            if src["split"] != manifest_split:
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


def sources_table(approved: pd.DataFrame, sdxl_manifest: pd.DataFrame,
                  by_image: dict[str, list[dict]]) -> pd.DataFrame:
    """Build ``sources.csv``: one row per approved SDXL source image.

    Args:
        approved: ``manifests/sdxl_sources_manifest.csv`` (``image_id, dataset, split``),
            the 249 real images sent to SDXL img2img.
        sdxl_manifest: the kept SDXL twins (to mark which sources produced one).
        by_image: output of :func:`records_by_image`.

    Returns:
        DataFrame ``id, dataset, split, label, source_image_path, twin_kept, record_ids``.
        ``label`` is ``supported`` (only real, camera images were used as sources).
    """
    kept = set(sdxl_manifest["image_id"])
    rows = []
    for row in approved.itertuples(index=False):
        recs = by_image[row.image_id]
        rows.append({
            "id": row.image_id,
            "dataset": row.dataset,
            "split": SPLIT_NAMES.get(row.split, row.split),
            "label": "supported",
            "source_image_path": recs[0]["image_path"],
            "twin_kept": row.image_id in kept,
            "record_ids": ";".join(r["id"] for r in recs),
        })
    return pd.DataFrame(rows)


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write records as JSON Lines (UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    """Command-line entry point: build vae.jsonl, sdxl.jsonl and sources.csv."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="only use the first N manifest rows and print instead of writing")
    args = ap.parse_args()

    by_image = {}
    for path in SOURCE_FILES.values():
        by_image.update(records_by_image(load_jsonl(path)))

    for method in ("vae", "sdxl"):
        manifest = pd.read_csv(VISION_COPIES / "manifests" / f"{method}_manifest.csv")
        if args.limit:
            manifest = manifest.head(args.limit)
        twins = twin_records(manifest, by_image)
        if args.limit:
            for t in twins[:3]:
                print(json.dumps(t, ensure_ascii=False)[:300])
            continue
        write_jsonl(twins, VISION_COPIES / f"{method}.jsonl")
        counts = pd.DataFrame(twins).groupby(["source", "split", "label"]).size()
        print(f"{method}: {len(manifest)} twin images -> {len(twins)} records\n"
              f"{counts.to_string()}\n")
        if method == "sdxl":
            approved = pd.read_csv(VISION_COPIES / "manifests" / "sdxl_sources_manifest.csv")
            table = sources_table(approved, manifest, by_image)
            table.to_csv(VISION_COPIES / "sources.csv", index=False)
            print(f"sources.csv: {len(table)} source images, "
                  f"{int(table['twin_kept'].sum())} with a kept twin")


if __name__ == "__main__":
    main()
