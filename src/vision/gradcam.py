from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.vision.detectors import (GenImageDataset, MEAN, STD, build_model, eval_transform, manifest_rows,
                                  target_layer)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXAMPLES = (("test", "real", 3), ("test", "Stable Diffusion v1.4", 3), ("validation", "BigGAN", 2))


def pick_examples(seed: int = 42) -> pd.DataFrame:
    parts = []
    for split, generator, n in EXAMPLES:
        rows = manifest_rows(split)
        parts.append(rows[rows["generator"] == generator].sample(n, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def grad_cam(model: torch.nn.Module, arch: str, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
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
    if arch == "vit_b16":
        act = act[:, 1:].transpose(1, 2).reshape(act.shape[0], -1, 14, 14)
        grad = grad[:, 1:].transpose(1, 2).reshape(grad.shape[0], -1, 14, 14)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    cam = cam / cam.flatten(1).max(dim=1).values.clamp(min=1e-8)[:, None, None]
    return cam.cpu().numpy(), torch.sigmoid(logits.detach()).cpu().numpy()


def unnormalise(x: torch.Tensor) -> np.ndarray:
    img = x.cpu() * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt-dir", default="models/F5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = pick_examples(args.seed)
    data = GenImageDataset(rows, args.root, eval_transform())
    x = torch.stack([data[i][0] for i in range(len(data))]).to(args.device)
    for arch in ("resnet50", "vit_b16"):
        ckpt = Path(args.ckpt_dir) / f"{arch}_s{args.seed}.pt"
        if not ckpt.exists():
            print(f"skip {arch}: {ckpt} not found")
            continue
        model = build_model(arch, pretrained=False).to(args.device)
        model.load_state_dict(torch.load(ckpt, map_location=args.device)["model"])
        cams, p_ai = grad_cam(model, arch, x)

        fig, axes = plt.subplots(2, 4, figsize=(12, 7.2))
        for i, ax in enumerate(axes.flat):
            ax.imshow(unnormalise(x[i]))
            ax.imshow(cams[i], cmap="jet", alpha=0.4)
            ax.set_title(f"{rows['generator'][i]} ({rows['split'][i]})\nP(AI) = {p_ai[i]:.2f}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Grad-CAM, {arch}")
        fig.tight_layout()
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(f"reports/figures/fig_gradcam_{arch}.png", dpi=300)
        plt.close(fig)

        Path("results/F5").mkdir(parents=True, exist_ok=True)
        table = rows[["image_id", "split", "generator", "class_name"]].assign(p_ai=p_ai)
        table.to_csv(f"results/F5/gradcam_{arch}.csv", index=False)
        print(f"{arch}: reports/figures/fig_gradcam_{arch}.png")


if __name__ == "__main__":
    main()
