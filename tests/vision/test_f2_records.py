"""Tests for the F2 shared-format converters (10-row fake data, no images needed).

Run: python -m pytest tests/vision -q
"""
import json

import pandas as pd

from src.eval.loaders import validate_record
from src.vision.load_verite import image_split, to_records
from src.vision.make_twin_records import m4fc_manifest, records_by_image, twin_records


def fake_verite(n_true: int = 10) -> pd.DataFrame:
    """VERITE-like rows: each true image has a true + a miscaptioned caption,
    and each false image one out-of-context caption (like the real file)."""
    rows = []
    for i in range(n_true):
        rows.append({"caption": f"real caption {i}", "image_path": f"images/true_{i}.jpg", "label": "true"})
        rows.append({"caption": f"wrong caption {i}", "image_path": f"images/true_{i}.jpg", "label": "miscaptioned"})
        rows.append({"caption": f"real caption {i}", "image_path": f"images/false_{i}.jpg", "label": "out-of-context"})
    return pd.DataFrame(rows)


def test_verite_records_are_valid_and_split_by_image():
    df = fake_verite()
    records = to_records(df, image_split(df))
    assert all(validate_record(r) == [] for r in records)
    assert records[0]["id"] == "verite_000000" and records[0]["label"] == "supported"
    assert records[1]["label"] == "contradicted" and records[1]["orig_label"] == "miscaptioned"
    # Both captions of one image always land in the same split (no image leakage).
    split_of = {}
    for r in records:
        assert split_of.setdefault(r["image_file"], r["split"]) == r["split"]


def test_twins_copy_every_record_of_the_image():
    df = fake_verite(n_true=10)
    records = to_records(df, image_split(df))
    split = records[0]["split"]
    manifest = pd.DataFrame([{"image_id": "true_0", "dataset": "VERITE", "split": split,
                              "method": "VAE", "twin_image_path": "twin/true_0_V.jpg",
                              "jpeg85_original_path": "orig/true_0.jpg", "similarity": 0.95}])
    twins = twin_records(manifest, records_by_image(records))
    assert [t["id"] for t in twins] == ["verite_000000_V", "verite_000001_V"]
    assert [t["label"] for t in twins] == ["supported", "contradicted"]
    assert all(t["image_path"] == "twin/true_0_V.jpg" for t in twins)
    assert all(validate_record(t) == [] for t in twins)


def test_m4fc_twins_take_the_source_split():
    records = [{"id": "m4fc_000010", "split": "test", "image_path": "m4fc_images/m4fc_000010.jpg", "label": "contradicted"},
               {"id": "m4fc_tc_000010", "split": "test", "image_path": "m4fc_images/m4fc_000010.jpg", "label": "supported"}]
    manifest = pd.DataFrame([{"image_id": "m4fc_000010", "dataset": "M4FC", "split": None, "method": "VAE",
                              "twin_image_path": "twin/m4fc_000010.jpg", "similarity": 0.92}])
    twins = twin_records(manifest, records_by_image(records))
    assert [t["id"] for t in twins] == ["m4fc_000010_V", "m4fc_tc_000010_V"]
    assert [(t["split"], t["label"]) for t in twins] == [("test", "contradicted"), ("test", "supported")]


def test_m4fc_twin_file_has_21_pairs():
    assert len(m4fc_manifest()) == 21
    twins = [json.loads(line) for line in open("data/copies/vision/m4fc_vae.jsonl", encoding="utf-8")]
    assert len(twins) == 42 and {t["split"] for t in twins} == {"test"}
    assert sorted(t["label"] for t in twins) == ["contradicted"] * 21 + ["supported"] * 21
    assert all(validate_record(t) == [] for t in twins)
