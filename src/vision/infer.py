"""Score one image with the F5 AI-image detector, for the demo (F11).

What it does
    Gives P(AI) for one picture with the ViT-B/16 detector from F5 (seed 42). The picture
    goes through exactly the F3 "debiased 96 px" input the detector was trained on:
      1. pixels only (EXIF orientation applied, format and metadata dropped), RGB;
      2. centre square crop, resize to 96 x 96 (bicubic);
      3. re-save as JPEG with the quantisation tables and chroma subsampling of a real
         training photo, picked from the F3 pool by a hash of the image key;
      4. eval_transform() from src/vision/detectors.py (resize 256, crop 224, ImageNet norm).
    F3 also turned some training images grey to balance the classes. That is a step over
    the whole dataset, so it is not applied to a single image.
    The F3 pool of JPEG settings was never saved. ``pool_from_cache`` recovers the exact
    ordered pool from the cached 96 px images (each one records its setting; only a pool
    of 500 fits all 13,000), and it is kept in git as models/F11/jpeg_pool.json (only
    quantisation tables, no image data). ``build_jpeg_pool`` rebuilds it from the raw
    training photos on Drive instead, the way F3 made it.
    Label: "AI-generated" if P(AI) >= 0.5 (the threshold in the report; never tuned on test).
    The claim verdict never uses this score (CLAUDE.md §3).

Inputs
    --ckpt       F5 checkpoint (default models/F5/vit_b16_s42.pt, Git LFS: run `git lfs pull`)
    --jpeg-pool  pool JSON (default models/F11/jpeg_pool.json)
    --image      a picture to score

Outputs
    check_image -> {"p_ai": float, "label": "AI-generated" | "Real", "model": str}
    --verify-cache / --verify-raw -> a CSV comparing our scores with results/F10.

How to run (repo root; `git lfs pull` first for the checkpoint)
    python -m src.vision.infer --pool-from-cache data/processed/vision/genimage_debiased96_jpeg.pkl
    python -m src.vision.infer --image photo.jpg
    python -m src.vision.infer --verify-cache data/processed/vision/genimage_debiased96_jpeg.pkl --n 10
    python -m src.vision.infer --verify-raw --drive /content/drive/MyDrive/TwinCheck --n 10   # needs Drive
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, JpegImagePlugin

SIZE = 96
THRESHOLD = 0.5
MODEL_NAME = "ViT-B/16 (F5, seed 42)"
MANIFESTS = Path("data/processed/vision")
DEFAULT_CKPT = "models/F5/vit_b16_s42.pt"
DEFAULT_POOL = "models/F11/jpeg_pool.json"
DEFAULT_CACHE = "data/processed/vision/genimage_debiased96_jpeg.pkl"
F10_PREDS = Path("results/F10/preds_vit_b16_s42_check.csv")


# --------------------------------------------------------------------------
# The F3 input
# --------------------------------------------------------------------------

def build_jpeg_pool(drive_root: str | Path, out_json: str | Path, seed: int = 42,
                    size: int = 500) -> list:
    """Rebuild F3's ordered pool of JPEG settings and save it as JSON.

    Same steps as the F3 / F5 notebooks: the three GenImage manifests in order, real
    RGB training photos, ``sample(size, random_state=seed)``, then keep settings with at
    least 2 tables and subsampling 0, 1 or 2. The order matters: an image's setting is
    ``pool[hash(key) % len(pool)]``.

    Returns:
        list of [tables, subsampling]; tables is a list of 64-number lists.
    """
    import pandas as pd

    gen = pd.concat([pd.read_csv(MANIFESTS / f"genimage_{s}_manifest.csv")
                     for s in ("train", "validation", "test")], ignore_index=True)
    rows = gen[(gen["split"] == "train") & (gen["class_name"] == "real")
               & (gen["original_mode"] == "RGB")].sample(size, random_state=seed)
    pool = []
    for path in rows["raw_export_path"]:
        with Image.open(Path(drive_root) / path) as im:
            tables = [list(t) for t in im.quantization.values()]
            sampling = JpegImagePlugin.get_sampling(im)
        if len(tables) >= 2 and sampling in (0, 1, 2):
            pool.append([tables, sampling])
    if not pool:
        raise ValueError("no usable JPEG settings found")
    Path(out_json).write_text(json.dumps(pool), encoding="utf-8")
    return pool


def pool_from_cache(cache_path: str | Path, out_json: str | Path | None = None) -> list:
    """Recover F3's ordered JPEG pool from the cached debiased images (no raw photos needed).

    Every cached image ``m`` was saved with ``pool[hash(m) % N]``, and its JPEG header
    records that setting. Only the true pool size N gives each slot a single setting for
    all 13,000 images; the cache's ``jpeg_settings`` field stores N (500). Raises if any
    image disagrees, so a returned pool reproduces every cached setting exactly.
    """
    import pickle

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    n = int(cache["jpeg_settings"])
    pool: list = [None] * n
    for md5, blob in zip(cache["md5"], cache["jpeg"]):
        with Image.open(io.BytesIO(blob)) as im:
            setting = [[list(t) for t in im.quantization.values()], JpegImagePlugin.get_sampling(im)]
        slot = int(hashlib.md5(md5.encode()).hexdigest()[:8], 16) % n
        if pool[slot] is None:
            pool[slot] = setting
        elif pool[slot] != setting:
            raise ValueError(f"cache does not fit a pool of {n}: slot {slot} has two settings")
    if any(s is None for s in pool):
        raise ValueError(f"{sum(s is None for s in pool)} pool slots never used in the cache")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(pool, separators=(",", ":")), encoding="utf-8")
    return pool


def load_jpeg_pool(path: str | Path) -> list:
    """The pool saved by ``build_jpeg_pool``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def square96(image: Image.Image) -> np.ndarray:
    """F3 ``load_square``: EXIF orientation, RGB, centre square crop, 96 x 96 bicubic."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    square = image.crop((left, top, left + s, top + s)).resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    return np.asarray(square)


def pixel_key(array: np.ndarray) -> str:
    """Key for an uploaded image: the md5 of its 96 px pixels (stable for the same picture)."""
    return hashlib.md5(array.tobytes()).hexdigest()


def encode_jpeg(array: np.ndarray, key: str, pool: list) -> bytes:
    """F3 ``encode``: JPEG with the pool setting chosen by the md5 of ``key``."""
    tables, sampling = pool[int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % len(pool)]
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="JPEG", qtables=tables, subsampling=sampling)
    return buf.getvalue()


def debiased_jpeg(image: Image.Image, pool: list, key: str | None = None) -> bytes:
    """Steps 1–3 of the F3 input. ``key`` is the GenImage md5 for training images;
    for an upload it defaults to ``pixel_key``."""
    array = square96(image)
    return encode_jpeg(array, key or pixel_key(array), pool)


def preprocess(image: Image.Image, pool: list, key: str | None = None):
    """Picture -> 1 x 3 x 224 x 224 tensor, exactly as the detector saw its inputs."""
    return blob_to_tensor(debiased_jpeg(image, pool, key))


def blob_to_tensor(blob: bytes):
    """Debiased JPEG bytes -> 1 x 3 x 224 x 224 tensor (step 4, eval_transform)."""
    from src.vision.detectors import eval_transform

    return eval_transform()(Image.open(io.BytesIO(blob)).convert("RGB")).unsqueeze(0)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def load_detector(ckpt_path: str | Path, arch: str = "vit_b16", device: str | None = None):
    """Load an F5 checkpoint (weights under "model") in eval mode."""
    import torch

    from src.vision.detectors import build_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    return model.to(device).eval()


def score_tensor(model, x) -> float:
    """P(AI) for one input tensor, scored as in F5/F10 (fp16 autocast on a GPU)."""
    import torch

    device = next(model.parameters()).device
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                enabled=device.type == "cuda"):
        logit = model(x.to(device)).reshape(-1)[0]
    return float(torch.sigmoid(logit.float()))


def label_for(p_ai: float) -> str:
    """'AI-generated' at or above the 0.5 threshold, else 'Real'."""
    return "AI-generated" if p_ai >= THRESHOLD else "Real"


def check_image(image: Image.Image, model, pool: list, blob: bytes | None = None) -> dict:
    """Score one picture: {"p_ai", "label", "model"}.

    ``blob``: an already debiased F3 JPEG (a cached example); it is scored as it is, which
    reproduces the F10 score exactly, instead of preprocessing ``image`` again.
    """
    p_ai = score_tensor(model, blob_to_tensor(blob) if blob else preprocess(image, pool))
    return {"p_ai": p_ai, "label": label_for(p_ai), "model": MODEL_NAME}


def gradcam_overlay(image: Image.Image, model, pool: list, arch: str = "vit_b16",
                    blob: bytes | None = None) -> Image.Image:
    """Grad-CAM heat map (src/vision/gradcam.py) over the 224 px input the detector saw.

    ``blob``: an already debiased F3 JPEG (cached example) to use instead of ``image``.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    from src.vision.gradcam import grad_cam, unnormalise

    device = next(model.parameters()).device
    x = (blob_to_tensor(blob) if blob else preprocess(image, pool)).to(device)
    cams, _ = grad_cam(model, arch, x)
    base = unnormalise(x[0])                               # H x W x 3 in [0, 1]
    heat = colormaps["jet"](cams[0])[..., :3]
    return Image.fromarray((255 * (0.6 * base + 0.4 * heat)).astype(np.uint8))


# --------------------------------------------------------------------------
# Checks against F10
# --------------------------------------------------------------------------

def sample_test_rows(n: int, seed: int = 42):
    """``n`` test rows from the F10 predictions, half real and half Stable Diffusion."""
    import pandas as pd

    preds = pd.read_csv(F10_PREDS)
    test = preds[preds["split"] == "test"]
    return test.groupby("class_name", group_keys=False).sample(n // 2, random_state=seed)


def verify_cache(model, cache_path: str | Path, n: int = 10) -> "pd.DataFrame":
    """Score ``n`` cached F3 inputs and compare with the F10 p_ai (same md5)."""
    import pandas as pd

    from src.vision.detectors import load_cache

    rows = sample_test_rows(n)
    blobs = load_cache(cache_path, list(rows["md5"]))
    ours = [score_tensor(model, blob_to_tensor(b)) for b in blobs]
    return pd.DataFrame({"md5": rows["md5"].to_numpy(), "class_name": rows["class_name"].to_numpy(),
                         "p_ai_f10": rows["p_ai"].to_numpy(), "p_ai_ours": ours,
                         "abs_diff": np.abs(np.array(ours) - rows["p_ai"].to_numpy())})


def verify_raw(model, pool: list, drive_root: str | Path, cache_path: str | Path | None,
               n: int = 10) -> "pd.DataFrame":
    """Raw GenImage files through ``debiased_jpeg`` (key = manifest md5) vs F10 / the cache.

    Uses only images that F3 did not turn grey, so the steps are the same.
    """
    import pandas as pd

    from src.vision.detectors import load_cache

    gen = pd.concat([pd.read_csv(MANIFESTS / f"genimage_{s}_manifest.csv")
                     for s in ("train", "validation", "test")], ignore_index=True)
    grey_csv = Path(drive_root) / "features/vision/genimage/clip_vitl14_debiased96/embedding_manifest.csv"
    if grey_csv.exists():
        grey = pd.read_csv(grey_csv)
        gen = gen[~grey["grayscale_added"].astype(bool).to_numpy()]
    rows = sample_test_rows(4 * n).merge(gen[["md5", "raw_export_path"]], on="md5").head(n)
    cached = load_cache(cache_path, list(rows["md5"])) if cache_path else [None] * len(rows)
    out = []
    for (_, r), blob_f3 in zip(rows.iterrows(), cached):
        with Image.open(Path(drive_root) / r["raw_export_path"]) as im:
            blob = debiased_jpeg(im, pool, key=r["md5"])
        p = score_tensor(model, blob_to_tensor(blob))
        out.append({"md5": r["md5"], "class_name": r["class_name"], "p_ai_f10": r["p_ai"], "p_ai_ours": p,
                    "abs_diff": abs(p - r["p_ai"]),
                    "same_bytes_as_cache": None if blob_f3 is None else blob == blob_f3})
    return pd.DataFrame(out)


def main() -> None:
    """CLI: build the pool, score one image, or run the checks against F10."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--jpeg-pool", default=DEFAULT_POOL)
    ap.add_argument("--image")
    ap.add_argument("--build-pool", action="store_true", help="rebuild from raw photos (needs --drive)")
    ap.add_argument("--pool-from-cache", metavar="PKL", help="recover the pool from the 96 px cache")
    ap.add_argument("--drive", help="Drive root, e.g. /content/drive/MyDrive/TwinCheck")
    ap.add_argument("--verify-cache", metavar="PKL")
    ap.add_argument("--verify-raw", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="results/F11/preprocess_check.csv")
    args = ap.parse_args()

    if args.build_pool or args.pool_from_cache:
        pool = (pool_from_cache(args.pool_from_cache, args.jpeg_pool) if args.pool_from_cache
                else build_jpeg_pool(args.drive, args.jpeg_pool))
        print(f"pool: {len(pool)} JPEG settings -> {args.jpeg_pool}")
        return
    pool, model = load_jpeg_pool(args.jpeg_pool), load_detector(args.ckpt)
    if args.image:
        with Image.open(args.image) as im:
            print(json.dumps(check_image(im, model, pool), indent=2))
        return
    import pandas as pd

    parts = []
    if args.verify_cache:
        parts.append(verify_cache(model, args.verify_cache, args.n).assign(check="cache"))
    if args.verify_raw:
        parts.append(verify_raw(model, pool, args.drive, args.verify_cache, args.n).assign(check="raw"))
    report = pd.concat(parts, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print(report.round(6).to_string(index=False))
    for check, part in report.groupby("check"):
        print(f"{check}: max |diff| = {part['abs_diff'].max():.2e}")


if __name__ == "__main__":
    main()
