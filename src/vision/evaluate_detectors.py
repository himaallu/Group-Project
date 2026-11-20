"""CNN vs ViT vs CLIP probe: seen/unseen generators, clean and corrupted (F5).

What it does
    * For every trained detector checkpoint (``train_detector.py``) and every
      corruption (clean, JPEG 50, JPEG 70, 20% crop), predicts the GenImage validation
      images of that setting's test generators.
    * For the CLIP probe (F3), embeds the same validation images under the same
      corruptions with CLIP ViT-L/14 and evaluates the probe of each setting (refit
      from the cached training embeddings; deterministic).
    * Writes one table with accuracy, AUROC and accuracy per generator.

Inputs
    --root: folder containing ``data/processed/vision/genimage/``.
    --ckpt-dir: checkpoints ``<arch>_<setting>_s<seed>.pt``.

Outputs
    results/F5/cnn_vit.csv   model, setting, corruption, n, accuracy, auroc, acc_<generator>

How to run (GPU)
    python -m src.vision.evaluate_detectors --root /content/genimage_root --ckpt-dir /content/drive/MyDrive/twincheck/models/F5
    Quick CPU check: --limit 4 --device cpu --skip-clip
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from src.vision.detectors import (CORRUPTIONS, GenImageDataset, build_model, corrupt,
                                  eval_transform, manifest_rows)
from src.vision.probe import SETTINGS, evaluate, load_genimage, subset, train_probe
from src.vision.train_detector import metrics, predict

CLIP_NAME = "openai/clip-vit-large-patch14"
CKPT_NAME = re.compile(r"^(?P<arch>resnet50|vit_b16)_(?P<setting>seen|unseen_biggan|unseen_sd)_s(?P<seed>\d+)$")


def detector_rows(ckpt_dir: Path, root: str, device: str, limit: int | None, batch: int) -> list[dict]:
    """Evaluate every checkpoint under every corruption."""
    rows = []
    for ckpt in sorted(ckpt_dir.glob("*.pt")):
        match = CKPT_NAME.match(ckpt.stem)
        if not match:
            continue
        arch, setting, seed = match["arch"], match["setting"], int(match["seed"])
        model = build_model(arch, pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        test = manifest_rows("validation", setting, "test", limit)
        for kind in CORRUPTIONS:
            loader = DataLoader(GenImageDataset(test, root, eval_transform(), kind), batch_size=batch,
                                num_workers=2)
            p_ai, y = predict(model, loader, device)
            rows.append({"model": arch, "setting": setting, "seed": seed, "corruption": kind,
                         **metrics(p_ai, y, test)})
            print(rows[-1]["model"], setting, kind, round(rows[-1]["accuracy"], 3))
    return rows


@torch.inference_mode()
def clip_embed(paths: list[Path], kind: str, model, processor, device: str, batch: int) -> np.ndarray:
    """L2-normalised CLIP ViT-L/14 image embeddings after a corruption (as in F2)."""
    feats = []
    for start in range(0, len(paths), batch):
        images = [corrupt(Image.open(p).convert("RGB"), kind) for p in paths[start:start + batch]]
        out = model.get_image_features(**processor(images=images, return_tensors="pt").to(device))
        out = out.pooler_output if hasattr(out, "pooler_output") else out
        feats.append(F.normalize(out.float(), dim=1).cpu().numpy())
    return np.concatenate(feats)


def clip_rows(root: str, device: str, limit: int | None, batch: int) -> list[dict]:
    """CLIP probe of each setting on the corrupted validation images."""
    X, meta = load_genimage()
    val = meta[meta["split"] == "validation"]
    if limit:
        val = val.groupby("generator", group_keys=False).head(limit)
    paths = [Path(root) / p for p in val["image_path"]]
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    rows = []
    for kind in CORRUPTIONS:
        emb = clip_embed(paths, kind, model, processor, device, batch)
        for setting, (train_gens, test_gens) in SETTINGS.items():
            tr = subset(meta, "train", train_gens, None)
            probe = train_probe(X[tr], (meta.loc[tr, "class_name"] == "fake").astype(int).to_numpy())
            keep = val["generator"].isin(test_gens).to_numpy()
            result = evaluate(probe, emb[keep], val[keep].reset_index(drop=True))
            rows.append({"model": "clip_probe", "setting": setting, "seed": 42, "corruption": kind,
                         **result})
            print("clip_probe", setting, kind, round(result["accuracy"], 3))
    return rows


def main() -> None:
    """Evaluate detectors and the CLIP probe; write results/F5/cnn_vit.csv."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt-dir", default="models/F5")
    ap.add_argument("--out", default="results/F5/cnn_vit.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-clip", action="store_true")
    args = ap.parse_args()

    rows = detector_rows(Path(args.ckpt_dir), args.root, args.device, args.limit, args.batch)
    if not args.skip_clip:
        rows += clip_rows(args.root, args.device, args.limit, args.batch)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
