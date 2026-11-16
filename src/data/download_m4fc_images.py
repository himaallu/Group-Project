"""Download M4FC images for the vision side of TwinCheck (task Y1, hand-off to Feba F2).

What it does
    Reads M4FC.json (UKPLab/M4FC), picks the rows to download (Arabic claims, the
    official test split, or everything), downloads each image from `image_url` and
    falls back to `wayback_image_url`, checks it opens with PIL, and saves it as an
    RGB JPEG named after the row's stable TwinCheck id (m4fc_000123.jpg).
    Re-running skips images that are already saved, so it is safe after a disconnect.

Inputs
    --json     path to M4FC.json (from `git clone https://github.com/UKPLab/M4FC`)
    --subset   arabic | test | all   (default: arabic)
    --limit    only the first N selected rows (use 10 for the test run)

Outputs
    <out_dir>/m4fc_000123.jpg          one file per image
    <out_dir>/manifest.csv             id, image_path, claim_language, split, status, url_used

How to run
    python src/data/download_m4fc_images.py --json data/raw/M4FC/data/M4FC.json \
        --out_dir data/processed/vision/m4fc_images --subset arabic --limit 10
"""

import argparse
import csv
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

MISSING = "not enough information"  # M4FC's value for empty fields
HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset download; TwinCheck student project)"}


def m4fc_id(idx):
    """Return the stable TwinCheck id for row `idx` of M4FC.json, e.g. m4fc_000123."""
    return f"m4fc_{idx:06d}"


def select_rows(rows, subset):
    """Pick which M4FC rows to download.

    Args:
        rows: list of dicts loaded from M4FC.json.
        subset: "arabic" (claim_language == Arabic), "test" (official test split,
            Arabic rows first) or "all" (Arabic first, then test, then the rest).
    Returns:
        list of (idx, row) tuples in download order.
    """
    indexed = list(enumerate(rows))
    arabic = [(i, r) for i, r in indexed if r.get("claim_language") == "Arabic"]
    if subset == "arabic":
        return arabic
    arabic_ids = {i for i, _ in arabic}
    test = [(i, r) for i, r in indexed if r.get("split") == "test" and i not in arabic_ids]
    if subset == "test":
        return arabic + test
    seen = arabic_ids | {i for i, _ in test}
    return arabic + test + [(i, r) for i, r in indexed if i not in seen]


def fetch_image(url, timeout=20):
    """Download one URL and return a PIL RGB image, or None if it fails or is not an image."""
    if not url or url == MISSING:
        return None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200 or len(resp.content) < 1000:
            return None
        img = Image.open(io.BytesIO(resp.content))
        img.load()  # forces a full decode, so broken files fail here
        return img.convert("RGB")
    except Exception:
        return None


def download_one(idx, row, out_dir):
    """Download one M4FC image (original URL, then Wayback copy) and save it as JPEG.

    Returns:
        dict for the manifest: id, image_path, claim_language, split, status, url_used.
        status is "ok", "exists" or "failed".
    """
    rid = m4fc_id(idx)
    path = os.path.join(out_dir, f"{rid}.jpg")
    record = {"id": rid, "image_path": path, "claim_language": row.get("claim_language"),
              "split": row.get("split"), "status": "failed", "url_used": ""}
    if os.path.exists(path) and os.path.getsize(path) > 0:
        record["status"] = "exists"
        return record
    for key in ("image_url", "wayback_image_url"):
        img = fetch_image(row.get(key))
        if img is not None:
            img.save(path, "JPEG", quality=95)
            record.update(status="ok", url_used=row.get(key))
            break
    return record


def main():
    """Parse arguments, download the selected images in parallel, write manifest.csv and print counts."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out_dir", default="data/processed/vision/m4fc_images")
    ap.add_argument("--subset", choices=["arabic", "test", "all"], default="arabic")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = json.load(open(args.json, encoding="utf-8"))
    todo = select_rows(rows, args.subset)[: args.limit]

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, i, r, args.out_dir) for i, r in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="images"):
            results.append(fut.result())
    results.sort(key=lambda r: r["id"])

    # Merge with an existing manifest so runs with different subsets add up.
    manifest = os.path.join(args.out_dir, "manifest.csv")
    old = {}
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            old = {r["id"]: r for r in csv.DictReader(f)}
    for r in results:
        if r["status"] == "exists" and r["id"] in old:
            continue  # keep the original "ok" row and its URL
        old[r["id"]] = r
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "image_path", "claim_language", "split", "status", "url_used"])
        w.writeheader()
        w.writerows(sorted(old.values(), key=lambda r: r["id"]))

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"subset={args.subset} selected={len(todo)} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    failed = [r["id"] for r in results if r["status"] == "failed"]
    if failed:
        print("failed ids (first 20):", failed[:20])


if __name__ == "__main__":
    main()
