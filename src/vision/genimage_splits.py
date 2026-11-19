import json
import random
from pathlib import Path

import pandas as pd

VISION = Path("data/processed/vision")
SEED = 42
N_TEST_REAL = 1000
GENERATORS = ["real", "BigGAN", "Stable Diffusion v1.4"]
FOLDERS = {"real": "real", "BigGAN": "fake/BigGAN", "Stable Diffusion v1.4": "fake/Stable_Diffusion_v1_4"}
COLUMNS = ["image_path", "md5", "label", "generator", "class_name", "split", "source_split", "exported_path", "source_shard"]


def load_pool():
    frames = [pd.read_csv(VISION / f"genimage_{name}_manifest.csv") for name in ["train", "validation", "test"]]
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["generator"].isin(GENERATORS)].copy()
    if "source_split" not in pool.columns:
        pool["source_split"] = pool["split"]
    pool["source_split"] = pool["source_split"].fillna(pool["split"])
    pool = pool.drop_duplicates("md5")
    pool["exported_path"] = [
        f"{VISION.as_posix()}/genimage/{row.source_split}/{FOLDERS[row.generator]}/{row.md5}.{row.image_path.rsplit('.', 1)[-1]}"
        for row in pool.itertuples()
    ]
    return pool


def select(pool, source_split, generator):
    rows = pool[(pool["source_split"] == source_split) & (pool["generator"] == generator)]
    return rows.sort_values("exported_path")


def build_splits(pool):
    real_train = select(pool, "train", "real")
    order = list(real_train.index)
    random.seed(SEED)
    random.shuffle(order)
    parts = {
        "train": [real_train.loc[order[N_TEST_REAL:]], select(pool, "train", "BigGAN")],
        "validation": [select(pool, "validation", "real"), select(pool, "validation", "BigGAN")],
        "test": [real_train.loc[order[:N_TEST_REAL]], select(pool, "validation", "Stable Diffusion v1.4")],
    }
    splits = {}
    for name, frames in parts.items():
        df = pd.concat(frames, ignore_index=True)
        df["split"] = name
        splits[name] = df[COLUMNS]
    return splits


def leakage_report(splits):
    hashes = {name: set(df["md5"]) for name, df in splits.items()}
    report = {}
    for name, df in splits.items():
        report[f"{name}_records"] = len(df)
        report[f"{name}_unique_hashes"] = len(hashes[name])
        report[f"{name}_counts"] = df.groupby("generator").size().to_dict()
    report["train_validation_hash_overlap"] = len(hashes["train"] & hashes["validation"])
    report["train_test_hash_overlap"] = len(hashes["train"] & hashes["test"])
    report["validation_test_hash_overlap"] = len(hashes["validation"] & hashes["test"])
    seen = set(splits["train"]["generator"]) | set(splits["validation"]["generator"])
    report["test_ai_generator_seen_in_training"] = bool(
        set(splits["test"].loc[splits["test"]["label"] == 1, "generator"]) & seen
    )
    report["leakage_detected"] = bool(
        report["train_validation_hash_overlap"]
        or report["train_test_hash_overlap"]
        or report["validation_test_hash_overlap"]
        or report["test_ai_generator_seen_in_training"]
        or any(report[f"{n}_records"] != report[f"{n}_unique_hashes"] for n in splits)
    )
    return report


def main():
    splits = build_splits(load_pool())
    for name, df in splits.items():
        df.to_csv(VISION / f"genimage_{name}_manifest.csv", index=False)
    report = leakage_report(splits)
    with open(VISION / "genimage_leakage_check_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    for name, df in splits.items():
        print(name, len(df), df.groupby("generator").size().to_dict())
    print("leakage_detected:", report["leakage_detected"])


if __name__ == "__main__":
    main()
