from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score

from src.vision.detectors import build_model, eval_transform, load_cache, manifest_rows


class ImageReader(CalibrationDataReader):
    def __init__(self, arrays: list[np.ndarray]):
        self.items = iter([{"image": a[None]} for a in arrays])

    def get_next(self):
        return next(self.items, None)


def to_arrays(blobs: list[bytes]) -> list[np.ndarray]:
    t = eval_transform()
    return [t(Image.open(io.BytesIO(b)).convert("RGB")).numpy() for b in blobs]


def export_fp32(arch: str, ckpt: str, path: Path) -> torch.nn.Module:
    model = build_model(arch, pretrained=False).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    if not path.exists():
        torch.onnx.export(model, torch.randn(1, 3, 224, 224), str(path), opset_version=17, input_names=["image"],
                          output_names=["logit"], dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
                          dynamo=False)
    return model


def time_ms(session, x: np.ndarray, warmup: int, runs: int) -> np.ndarray:
    for _ in range(warmup):
        session.run(None, {"image": x})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {"image": x})
        times.append((time.perf_counter() - start) * 1000)
    return np.array(times)


def scores(session, arrays: list[np.ndarray]) -> np.ndarray:
    logits = np.concatenate([session.run(None, {"image": np.stack(arrays[s:s + 50])})[0][:, 0]
                             for s in range(0, len(arrays), 50)])
    return 1 / (1 + np.exp(-logits))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="results/F10")
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--check", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    work, out = Path(args.work), Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.ckpt).stem

    val = manifest_rows("validation", args.limit)
    calib_rows = val.groupby("class_name", group_keys=False).sample(args.calib // 2, random_state=args.seed)
    test = manifest_rows("test", args.limit)
    check_idx = np.sort(test.groupby("class_name", group_keys=False).sample(args.check // 2, random_state=args.seed).index.to_numpy())
    check_rows = test.iloc[check_idx]
    y = (check_rows["class_name"] == "fake").to_numpy().astype(int)
    calib = to_arrays(load_cache(args.cache, calib_rows["md5"]))
    check = to_arrays(load_cache(args.cache, check_rows["md5"]))

    fp32, prep, int8 = work / f"{name}.onnx", work / f"{name}_prep.onnx", work / f"{name}_int8_static.onnx"
    model = export_fp32(args.arch, args.ckpt, fp32)
    quant_pre_process(str(fp32), str(prep))
    quantize_static(str(prep), str(int8), ImageReader(calib), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)

    probe = torch.from_numpy(np.stack(check[:4]))
    with torch.no_grad():
        reference = model(probe).numpy()
        torch_p = torch.sigmoid(torch.cat([model(torch.from_numpy(np.stack(check[s:s + 50])))
                                           for s in range(0, len(check), 50)])[:, 0]).numpy()
    one = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
    rows = []
    for precision, path in (("fp32", fp32), ("int8 static per-channel", int8)):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        p = scores(session, check)
        cpu = time_ms(session, one, args.warmup, args.runs)
        rows.append({"arch": args.arch, "checkpoint": name, "precision": precision,
                     "onnx_mb": round(os.path.getsize(path) / 1e6, 1),
                     "max_abs_diff": float(np.abs(session.run(None, {"image": probe.numpy()})[0] - reference).max()),
                     "ort_cpu_ms_mean": float(cpu.mean()), "ort_cpu_ms_p50": float(np.median(cpu)),
                     "cpu_threads": os.cpu_count(), "calibration_images": len(calib), "test_images": len(check),
                     "test_accuracy": accuracy_score(y, p >= 0.5), "test_auroc": roc_auc_score(y, p),
                     "same_decision_as_pytorch": float(((p >= 0.5) == (torch_p >= 0.5)).mean()),
                     "score_correlation_with_pytorch": float(np.corrcoef(p, torch_p)[0, 1])})
        print(json.dumps(rows[-1]), flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / f"onnx_int8_static_{name}.csv", index=False)


if __name__ == "__main__":
    main()
