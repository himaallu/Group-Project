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
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return np.array(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="models/F5")
    ap.add_argument("--onnx-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    onnx_dir = Path(args.onnx_dir or args.ckpt_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)

    rows = []
    for arch in ("resnet50", "vit_b16"):
        ckpt = Path(args.ckpt_dir) / f"{arch}_s{args.seed}.pt"
        if not ckpt.exists():
            print(f"skip {arch}: {ckpt} not found")
            continue
        model = build_model(arch, pretrained=False).eval()
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        path = onnx_dir / f"{arch}_s{args.seed}.onnx"
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
