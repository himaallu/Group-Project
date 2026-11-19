"""Grad-CAM heatmaps for the F5 detectors (8 GenImage validation examples).

What it does
    For the ``seen`` ResNet-50 and ViT-B/16 checkpoints, computes Grad-CAM for the
    AI logit on 8 validation images (3 real, 3 SD v1.4, 2 BigGAN; sampled with seed 42).
    ResNet-50: activations of the last bottleneck block (7x7). ViT-B/16: patch tokens at the
    input LayerNorm of the last encoder block, reshaped to 14x14 (the CLS token is dropped). Heatmaps are
    upsampled to 224x224 and drawn over the input crop.

Inputs
    --root (GenImage images), --ckpt-dir (``<arch>_seen_s42.pt``).

Outputs
    reports/figures/fig_gradcam_<arch>.png       8 images, heatmap overlay, label and P(AI)
    results/F5/gradcam_<arch>.csv                image_id, generator, label, p_ai

How to run (GPU or CPU)
    python -m src.vision.gradcam --root /content/genimage_root --ckpt-dir /content/drive/MyDrive/twincheck/models/F5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.vision.detectors import GenImageDataset, MEAN, STD, build_model, eval_transform, target_layer
from src.vision.probe import CLIP_MANIFEST

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

PER_GENERATOR = {"real": 3, "Stable Diffusion v1.4": 3, "BigGAN": 2}


def pick_examples(seed: int = 42) -> pd.DataFrame:
    """8 validation images: 3 real, 3 SD v1.4, 2 BigGAN (seeded sample)."""
    meta = pd.read_csv(CLIP_MANIFEST)
    val = meta[meta["split"] == "validation"]
    parts = [val[val["generator"] == g].sample(n, random_state=seed) for g, n in PER_GENERATOR.items()]
    return pd.concat(parts).reset_index(drop=True)


def grad_cam(model: torch.nn.Module, arch: str, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM maps (n, 224, 224) in [0, 1] and P(AI) for a batch of images."""
    store = {}
    layer = target_layer(model, arch)
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    model.eval()
    model.zero_grad()
    logits = model(x).squeeze(1)
    logits.sum().backward()
    h1.remove()
    h2.remove()
    act, grad = store["act"].detach(), store["grad"].detach()
    if arch == "vit_b16":  # (n, 197, 768) -> drop CLS -> (n, 768, 14, 14)
        act = act[:, 1:].transpose(1, 2).reshape(act.shape[0], -1, 14, 14)
        grad = grad[:, 1:].transpose(1, 2).reshape(grad.shape[0], -1, 14, 14)
    weights = grad.mean(dim=(2, 3), keepdim=True)           # one weight per channel
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    cam = cam / cam.flatten(1).max(dim=1).values.clamp(min=1e-8)[:, None, None]
    return cam.cpu().numpy(), torch.sigmoid(logits.detach()).cpu().numpy()


def unnormalise(x: torch.Tensor) -> np.ndarray:
    """Normalised tensor (3, H, W) -> RGB array in [0, 1] for plotting."""
    img = x.cpu() * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def main() -> None:
    """Grad-CAM figure + table for each seen checkpoint found."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt-dir", default="models/F5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = pick_examples(args.seed)
    data = GenImageDataset(rows, args.root, eval_transform())
    x = torch.stack([data[i][0] for i in range(len(data))]).to(args.device)
    for arch in ("resnet50", "vit_b16"):
        ckpt = Path(args.ckpt_dir) / f"{arch}_seen_s{args.seed}.pt"
        if not ckpt.exists():
            print(f"skip {arch}: {ckpt} not found")
            continue
        model = build_model(arch, pretrained=False).to(args.device)
        model.load_state_dict(torch.load(ckpt, map_location=args.device)["model"])
        cams, p_ai = grad_cam(model, arch, x)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
        for i, ax in enumerate(axes.flat):
            ax.imshow(unnormalise(x[i]))
            ax.imshow(cams[i], cmap="jet", alpha=0.4)
            ax.set_title(f"{rows['generator'][i]}\nP(AI) = {p_ai[i]:.2f}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Grad-CAM, {arch} (seen setting)")
        fig.tight_layout()
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(f"reports/figures/fig_gradcam_{arch}.png", dpi=300)
        plt.close(fig)

        Path("results/F5").mkdir(parents=True, exist_ok=True)
        table = rows[["image_id", "generator", "class_name"]].assign(p_ai=p_ai)
        table.to_csv(f"results/F5/gradcam_{arch}.csv", index=False)
        print(f"{arch}: reports/figures/fig_gradcam_{arch}.png")


if __name__ == "__main__":
    main()
