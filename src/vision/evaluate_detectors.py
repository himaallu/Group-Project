from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from src.vision.detectors import CORRUPTIONS, GenImageDataset, build_model, corrupt, eval_transform, manifest_rows
from src.vision.train_detector import metrics, predict

CLIP_NAME = "openai/clip-vit-large-patch14"
CLIP_EMB = Path("features/vision/genimage/genimage_clip_vit_l14_image_projected_embeddings_float16.npy")
CLIP_MANIFEST = Path("features/vision/genimage/genimage_image_embedding_manifest.csv")
CKPT_NAME = re.compile(r"^(?P<arch>resnet50|vit_b16)_s(?P<seed>\d+)$")
EVAL_SPLITS = ("validation", "test")


def detector_rows(ckpt_dir: Path, root: str, device: str, limit: int | None, batch: int) -> list[dict]:
    rows = []
    for ckpt in sorted(ckpt_dir.glob("*.pt")):
        match = CKPT_NAME.match(ckpt.stem)
        if not match:
            continue
        arch, seed = match["arch"], int(match["seed"])
        model = build_model(arch, pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        for split in EVAL_SPLITS:
            data = manifest_rows(split, limit)
            for kind in CORRUPTIONS:
                loader = DataLoader(GenImageDataset(data, root, eval_transform(), kind), batch_size=batch,
                                    num_workers=2)
                p_ai, y = predict(model, loader, device)
                rows.append({"model": arch, "seed": seed, "split": split, "corruption": kind,
                             **metrics(p_ai, y, data)})
                print(arch, seed, split, kind, round(rows[-1]["accuracy"], 3))
    return rows


@torch.inference_mode()
def clip_embed(paths: list[Path], kind: str, model, processor, device: str, batch: int) -> np.ndarray:
    feats = []
    for start in range(0, len(paths), batch):
        images = [corrupt(Image.open(p).convert("RGB"), kind) for p in paths[start:start + batch]]
        out = model.get_image_features(**processor(images=images, return_tensors="pt").to(device))
        out = out.pooler_output if hasattr(out, "pooler_output") else out
        feats.append(F.normalize(out.float(), dim=1).cpu().numpy())
    return np.concatenate(feats)


def fit_clip_probe() -> LogisticRegression:
    emb = np.load(CLIP_EMB).astype(np.float32)
    index = {image_id: i for i, image_id in enumerate(pd.read_csv(CLIP_MANIFEST)["image_id"])}
    train = manifest_rows("train")
    X = emb[train["image_id"].map(index).to_numpy()]
    y = (train["class_name"] == "fake").astype(int).to_numpy()
    return LogisticRegression(max_iter=2000, random_state=42).fit(X, y)


def clip_rows(root: str, device: str, limit: int | None, batch: int) -> list[dict]:
    probe = fit_clip_probe()
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    rows = []
    for split in EVAL_SPLITS:
        data = manifest_rows(split, limit)
        paths = [Path(root) / p for p in data["image_path"]]
        y = (data["class_name"] == "fake").astype(int).to_numpy()
        for kind in CORRUPTIONS:
            p_ai = probe.predict_proba(clip_embed(paths, kind, model, processor, device, batch))[:, 1]
            rows.append({"model": "clip_probe", "seed": 42, "split": split, "corruption": kind,
                         **metrics(p_ai, y, data)})
            print("clip_probe", split, kind, round(rows[-1]["accuracy"], 3))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
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
