"""JPEG-85 format control for the authenticity probe (CLIP ViT-L/14 embeddings).

Why
    In our GenImage subset every fake image is PNG (BigGAN 128x128, SD v1.4 512x512)
    and every real image is JPEG. A probe can then score well by spotting JPEG
    compression instead of AI artefacts. Our image twins are all JPEG 85, so this
    matters for the leakage test.

What it does
    * Samples N validation images per generator (real, SD v1.4, BigGAN), seed 42.
    * Embeds each image twice with CLIP ViT-L/14 (same model and normalisation as F2):
      ``unchanged`` (file as is) and ``jpeg85`` (re-saved in memory as JPEG quality 85).
    * Checks that ``unchanged`` reproduces the stored F2 embedding (cosine ~ 1), so
      the two variants are comparable with the probe's training features.

Inputs
    features/vision/genimage/genimage_image_embedding_manifest.csv (+ float16 embeddings)
    The GenImage image files (not in git): pass the folder(s) that contain ``data/processed/vision/genimage``.

Outputs
    features/vision/genimage/jpeg85_control_clip.npy       (2N*3, 768) float16
    features/vision/genimage/jpeg85_control_manifest.csv   image_id, generator, class_name, variant, cos_to_stored

How to run (CPU: about 0.2 s per image; GPU much faster)
    python -m src.vision.jpeg_control --image-roots /content/drive/MyDrive/twincheck --n 300
    python -m src.vision.jpeg_control --image-roots <root> --n 10      # quick check
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.vision.probe import CLIP_EMB, CLIP_MANIFEST, JPEG_CONTROL_EMB, JPEG_CONTROL_MANIFEST

CLIP_NAME = "openai/clip-vit-large-patch14"
SEED = 42


def find_image(rel_path: str, roots: list[Path]) -> Path:
    """Return the first existing ``root / rel_path`` (the Drive download is split in parts)."""
    for root in roots:
        if (root / rel_path).is_file():
            return root / rel_path
    raise FileNotFoundError(rel_path)


def to_jpeg85(image: Image.Image) -> Image.Image:
    """Re-save an RGB image as JPEG quality 85 in memory and decode it again."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


@torch.inference_mode()
def embed(images: list[Image.Image], model: CLIPModel, processor: CLIPProcessor) -> np.ndarray:
    """L2-normalised CLIP image embeddings (projected, 768-d), as in F2."""
    inputs = processor(images=images, return_tensors="pt")
    out = model.get_image_features(**inputs)
    # transformers >= 5 returns an output object; older versions return the tensor.
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return F.normalize(feats.float(), p=2, dim=1).numpy()


def main() -> None:
    """Sample, embed both variants, check against stored embeddings, save."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image-roots", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=300, help="images per generator")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    roots = [Path(r) for r in args.image_roots]

    meta = pd.read_csv(CLIP_MANIFEST)
    stored = np.load(CLIP_EMB).astype(np.float32)
    val = meta[meta["split"] == "validation"]
    sample = val.groupby("generator", group_keys=False).sample(n=args.n, random_state=SEED)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = CLIPModel.from_pretrained(CLIP_NAME).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)

    feats, rows = [], []
    for start in range(0, len(sample), args.batch):
        part = sample.iloc[start:start + args.batch]
        images = [Image.open(find_image(p, roots)).convert("RGB") for p in part["image_path"]]
        for variant, batch in (("unchanged", images), ("jpeg85", [to_jpeg85(im) for im in images])):
            emb = embed(batch, model, processor)
            feats.append(emb)
            for (row_idx, row), vec in zip(part.iterrows(), emb):
                rows.append({"image_id": row["image_id"], "generator": row["generator"],
                             "class_name": row["class_name"], "variant": variant,
                             "cos_to_stored": float(vec @ stored[row_idx])})
        print(f"{start + len(part)}/{len(sample)} images")

    manifest = pd.DataFrame(rows)
    np.save(JPEG_CONTROL_EMB, np.concatenate(feats).astype(np.float16))
    manifest.to_csv(JPEG_CONTROL_MANIFEST, index=False)
    print(manifest.groupby("variant")["cos_to_stored"].describe()[["mean", "min"]])


if __name__ == "__main__":
    main()
