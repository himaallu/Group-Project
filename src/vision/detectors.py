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

SPLITS_DIR = Path("data/processed/vision")
SPLITS = ("train", "validation", "test")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CORRUPTIONS = ("clean", "jpeg50", "jpeg70", "crop20")


class RandomJPEG:
    def __init__(self, low: int = 60, high: int = 95):
        self.low, self.high = low, high

    def __call__(self, image: Image.Image) -> Image.Image:
        return jpeg(image, random.randint(self.low, self.high))


class RandomBlur:
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))
        return image


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def corrupt(image: Image.Image, kind: str) -> Image.Image:
    if kind == "clean":
        return image
    if kind in ("jpeg50", "jpeg70"):
        return jpeg(image, int(kind[4:]))
    if kind == "crop20":
        w, h = image.size
        dw, dh = int(w * 0.1), int(h * 0.1)
        return image.crop((dw, dh, w - dw, h - dh))
    raise ValueError(kind)


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandomJPEG(60, 95),
        RandomBlur(0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def manifest_rows(split: str, limit: int | None = None) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(split)
    rows = pd.read_csv(SPLITS_DIR / f"genimage_{split}_manifest.csv")
    rows = rows.assign(image_path=rows["raw_export_path"], image_id=rows["md5"])
    if limit:
        rows = rows.groupby("generator", group_keys=False).head(limit)
    return rows.reset_index(drop=True)


class GenImageDataset(Dataset):
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
    return model.layer4[-1] if arch == "resnet50" else model.encoder.layers[-1].ln_1
