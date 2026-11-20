"""Export the F5 detectors to ONNX and measure milliseconds per image.

What it does
    For each ``seen`` checkpoint (ResNet-50, ViT-B/16): exports to ONNX (opset 17,
    dynamic batch), checks that ONNX Runtime gives the same logits as PyTorch on 4 random
    inputs, and times one image at a time (batch 1, 224x224): ONNX Runtime on CPU and,
    if a GPU is present, PyTorch on GPU. 10 warm-up runs, then 100 timed runs.

Inputs
    --ckpt-dir with ``<arch>_seen_s42.pt``.

Outputs
    <onnx-dir>/<arch>_seen.onnx       (~100 MB ResNet-50, ~330 MB ViT-B/16: keep on Drive, not in git)
    results/F5/onnx_latency.csv       arch, onnx_mb, max_abs_diff, ort_cpu_ms_mean, ort_cpu_ms_p50,
                                      torch_gpu_ms_mean, cpu, gpu

How to run
    python -m src.vision.export_onnx --ckpt-dir /content/drive/MyDrive/twincheck/models/F5
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from src.vision.detectors import build_model

WARMUP, RUNS = 10, 100


def time_ms(fn) -> np.ndarray:
    """Milliseconds of ``fn()`` over RUNS calls after WARMUP calls."""
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return np.array(times)


def main() -> None:
    """Export, verify and time each seen checkpoint."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", default="models/F5")
    ap.add_argument("--onnx-dir", default=None, help="default: same as --ckpt-dir")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    onnx_dir = Path(args.onnx_dir or args.ckpt_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    # The fused attention fast path (aten::_native_multi_head_attention) used by ViT in
    # eval mode has no ONNX export; the plain path gives the same outputs.
    torch.backends.mha.set_fastpath_enabled(False)

    rows = []
    for arch in ("resnet50", "vit_b16"):
        ckpt = Path(args.ckpt_dir) / f"{arch}_seen_s{args.seed}.pt"
        if not ckpt.exists():
            print(f"skip {arch}: {ckpt} not found")
            continue
        model = build_model(arch, pretrained=False).eval()
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        path = onnx_dir / f"{arch}_seen.onnx"
        torch.onnx.export(model, torch.randn(1, 3, 224, 224), str(path), opset_version=17,
                          input_names=["image"], output_names=["logit"],
                          dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}}, dynamo=False)

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        check = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            reference = model(check).numpy()
        diff = float(np.abs(session.run(None, {"image": check.numpy()})[0] - reference).max())

        one = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
        cpu_ms = time_ms(lambda: session.run(None, {"image": one}))
        gpu_ms, gpu_name = None, None
        if torch.cuda.is_available():
            gpu_model, x = model.cuda(), torch.from_numpy(one).cuda()

            def gpu_call():
                with torch.no_grad():
                    gpu_model(x)
                torch.cuda.synchronize()
            gpu_ms, gpu_name = float(time_ms(gpu_call).mean()), torch.cuda.get_device_name(0)

        rows.append({"arch": arch, "onnx_mb": round(path.stat().st_size / 1e6, 1), "max_abs_diff": diff,
                     "ort_cpu_ms_mean": float(cpu_ms.mean()), "ort_cpu_ms_p50": float(np.median(cpu_ms)),
                     "torch_gpu_ms_mean": gpu_ms, "cpu": platform.processor() or platform.machine(),
                     "gpu": gpu_name})
        print(rows[-1])

    Path("results/F5").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv("results/F5/onnx_latency.csv", index=False)


if __name__ == "__main__":
    main()
