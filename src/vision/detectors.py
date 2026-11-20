"""Shared parts of the F5 AI-image detectors: data, augmentation and the two models.

What it contains
    * ``GenImageDataset``: GenImage images listed in the F2 manifest
      (``features/vision/genimage/genimage_image_embedding_manifest.csv``), label 1 = AI.
    * ``train_transform`` / ``eval_transform``: augmentation from CLAUDE.md §9 F5
      (random resized crop, horizontal flip, JPEG quality 60-95, Gaussian blur) and the
      plain evaluation transform (resize 256, centre crop 224). The random JPEG step is
      applied to real and AI images alike, so JPEG compression is no longer a cue
      (every GenImage fake here is PNG, every real is JPEG; see F3).
    * ``corrupt``: test-time corruptions for the robustness table (JPEG 50/70, 20% crop).
    * ``build_model``: ImageNet-pretrained ResNet-50 or ViT-B/16 with one output logit and
      only the **last 4 blocks** (+ head) trainable: ResNet-50 = last block of layer3 and
      the 3 blocks of layer4; ViT-B/16 = the last 4 encoder layers + final LayerNorm.
    * ``SETTINGS``: seen / unseen-generator splits, the same as the F3 CLIP probe.

Inputs / outputs
    Used by ``train_detector.py``, ``evaluate_detectors.py``, ``gradcam.py`` and
    ``export_onnx.py``; no files of its own.
"""
from __future__ import annotations

import io
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

from src.vision.probe import CLIP_MANIFEST, SETTINGS

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics
CORRUPTIONS = ("clean", "jpeg50", "jpeg70", "crop20")


class RandomJPEG:
    """Re-save the image as JPEG with a random quality in [low, high]."""

    def __init__(self, low: int = 60, high: int = 95):
        self.low, self.high = low, high

    def __call__(self, image: Image.Image) -> Image.Image:
        return jpeg(image, random.randint(self.low, self.high))


class RandomBlur:
    """Gaussian blur with probability p and a random radius in [0.1, 2.0]."""

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))
        return image


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Re-save an image as JPEG of the given quality (in memory)."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def corrupt(image: Image.Image, kind: str) -> Image.Image:
    """Test-time corruption: clean, JPEG quality 50 / 70, or a centre crop keeping 80%."""
    if kind == "clean":
        return image
    if kind in ("jpeg50", "jpeg70"):
        return jpeg(image, int(kind[4:]))
    if kind == "crop20":
        w, h = image.size
        dw, dh = int(w * 0.1), int(h * 0.1)  # 10% off each side = 20% of each dimension
        return image.crop((dw, dh, w - dw, h - dh))
    raise ValueError(kind)


def train_transform():
    """Training augmentation: crop, flip, JPEG 60-95, blur, then normalise."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandomJPEG(60, 95),
        RandomBlur(0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    """Evaluation transform: resize 256, centre crop 224, normalise."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def manifest_rows(split: str, setting: str, side: str, limit: int | None = None) -> pd.DataFrame:
    """Rows of the GenImage manifest for one split of a seen/unseen setting.

    Args:
        split: ``train`` or ``validation``.
        setting: a key of ``SETTINGS`` (seen, unseen_biggan, unseen_sd).
        side: ``train`` (training generators) or ``test`` (test generators).
        limit: keep only N rows per generator (quick checks).
    """
    meta = pd.read_csv(CLIP_MANIFEST)
    gens = SETTINGS[setting][0 if side == "train" else 1]
    rows = meta[(meta["split"] == split) & meta["generator"].isin(gens)]
    if limit:
        rows = rows.groupby("generator", group_keys=False).head(limit)
    return rows.reset_index(drop=True)


class GenImageDataset(Dataset):
    """GenImage images from manifest rows; returns (tensor, label, row index)."""

    def __init__(self, rows: pd.DataFrame, root: str | Path, transform, corruption: str = "clean"):
        self.rows, self.root = rows, Path(root)
        self.transform, self.corruption = transform, corruption

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        image = Image.open(self.root / row["image_path"]).convert("RGB")
        image = corrupt(image, self.corruption)
        label = torch.tensor(float(row["class_name"] == "fake"))
        return self.transform(image), label, i


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    """ResNet-50 or ViT-B/16 with a 1-logit head and only the last 4 blocks trainable."""
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        trainable = [model.layer3[-1], model.layer4, model.fc]
    elif arch == "vit_b16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.heads = nn.Sequential(nn.Linear(model.hidden_dim, 1))
        trainable = [*model.encoder.layers[-4:], model.encoder.ln, model.heads]
    else:
        raise ValueError(f"unknown arch {arch}")
    for p in model.parameters():
        p.requires_grad = False
    for module in trainable:
        for p in module.parameters():
            p.requires_grad = True
    return model


def target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Layer whose activations Grad-CAM uses (last conv block / input LayerNorm of the last encoder block).

    The output of the last ViT block reaches the head only through the CLS token, so its
    patch tokens get zero gradient; ``ln_1`` of that block still sees all tokens.
    """
    return model.layer4[-1] if arch == "resnet50" else model.encoder.layers[-1].ln_1
