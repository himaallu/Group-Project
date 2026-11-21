"""Convert VERITE (image + caption pairs) to the TwinCheck shared record format.

What it does
    * Reads the VERITE rows (``caption``, ``image_path``, ``label``) in their original
      order, from the F1 manifest, the authors' ``VERITE.csv`` or a JSON Lines file.
    * Maps the labels: ``true`` -> ``supported``; ``miscaptioned`` and
      ``out-of-context`` -> ``contradicted`` (the caption does not describe the photo).
      The original label stays in ``orig_label``.
    * Uses the F1 split when the input has a ``split`` column
      (``data/processed/vision/verite_manifest.csv``: split by image, 704 / 151 / 146
      records). Without it, splits images 70 / 15 / 15, stratified by label set, seed 42.
    * Writes one shared-format record per row, ids ``verite_000000`` ... in row order.

Inputs
    * ``--input``: ``data/processed/vision/verite_manifest.csv`` (or ``VERITE.csv`` /
      a ``.jsonl`` with the same keys).
    * ``--image-root``: folder (repo-relative) the VERITE ``images/...`` paths live in.

Outputs
    * ``data/processed/vision/verite.jsonl`` (shared format, CLAUDE.md §4), plus the
      extra field ``image_file`` (e.g. ``true_0.jpg``) used to join image embeddings
      and image twins.

How to run
    python -m src.vision.load_verite --input data/processed/vision/verite_manifest.csv
    python -m src.vision.load_verite --input data/processed/vision/verite_manifest.csv --limit 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {
    "true": "supported",
    "miscaptioned": "contradicted",
    "out-of-context": "contradicted",
}
DEFAULT_IMAGE_ROOT = "data/raw/vision/verite/images_extracted"
SPLIT_NAMES = {"validation": "val"}
DEFAULT_OUTPUT = "data/processed/vision/verite.jsonl"
SEED = 42


def read_rows(path: str | Path) -> pd.DataFrame:
    """Read VERITE rows from a CSV or JSON Lines file.

    Args:
        path: ``VERITE.csv`` or a ``.jsonl`` file with ``caption``, ``image_path``, ``label``.

    Returns:
        DataFrame with columns ``caption``, ``image_path``, ``label`` in file order.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    missing = {"caption", "image_path", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    keep = ["caption", "image_path", "label"] + (["split"] if "split" in df.columns else [])
    return df[keep].reset_index(drop=True)


def manifest_split(df: pd.DataFrame) -> dict[str, str]:
    """Image file name -> split, from the ``split`` column of the F1 manifest."""
    return {Path(p).name: SPLIT_NAMES.get(s, s) for p, s in zip(df["image_path"], df["split"])}


def image_split(df: pd.DataFrame, seed: int = SEED) -> dict[str, str]:
    """Split images 70 / 15 / 15, stratified by each image's set of labels.

    Same procedure as the F1 notebook, so the split is reproducible.

    Args:
        df: VERITE rows with ``image_path`` and ``label``.
        seed: random seed for both ``train_test_split`` calls.

    Returns:
        Mapping image file name (e.g. ``true_0.jpg``) -> ``train`` / ``val`` / ``test``.
    """
    image_file = df["image_path"].map(lambda p: Path(p).name)
    # One row per image; the stratum is the sorted set of labels the image appears with
    # (true images appear as "miscaptioned|true", false images as "out-of-context").
    groups = (df.assign(image_file=image_file)
                .groupby("image_file")["label"]
                .agg(lambda x: "|".join(sorted(set(x))))
                .reset_index())
    train, temp = train_test_split(groups, test_size=0.30, random_state=seed,
                                   stratify=groups["label"])
    val, test = train_test_split(temp, test_size=0.50, random_state=seed,
                                 stratify=temp["label"])
    split = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        split.update({f: name for f in part["image_file"]})
    return split


def to_records(df: pd.DataFrame, split: dict[str, str],
               image_root: str = DEFAULT_IMAGE_ROOT) -> list[dict]:
    """Turn VERITE rows into shared-format records.

    Args:
        df: VERITE rows (``caption``, ``image_path``, ``label``) in original order.
        split: image file name -> split, from :func:`image_split`.
        image_root: repo-relative folder that contains VERITE's ``images/`` folder.

    Returns:
        List of records with the CLAUDE.md §4 keys plus ``image_file``.
    """
    records = []
    for i, row in df.iterrows():
        orig = str(row["label"]).strip().lower()
        image_file = Path(row["image_path"]).name
        records.append({
            "id": f"verite_{i:06d}",
            "source": "verite",
            "split": split[image_file],
            "language": "en",
            "claim": str(row["caption"]).strip(),
            "evidence": [],  # VERITE has no text evidence; the photo is the evidence
            "image_path": f"{image_root}/images/{image_file}",
            "label": LABEL_MAP[orig],
            "orig_label": orig,
            "image_file": image_file,
        })
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write records as JSON Lines (UTF-8, one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    """Command-line entry point: read, split, convert, write, print counts."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--limit", type=int, default=None,
                    help="only write the first N records (the split still uses all rows)")
    args = ap.parse_args()

    df = read_rows(args.input)
    split = manifest_split(df) if "split" in df.columns else image_split(df)
    records = to_records(df, split, args.image_root)
    if args.limit:
        records = records[:args.limit]
    write_jsonl(records, args.output)

    counts = pd.DataFrame(records).groupby(["split", "label"]).size()
    print(f"wrote {len(records)} records -> {args.output}\n{counts.to_string()}")


if __name__ == "__main__":
    main()
