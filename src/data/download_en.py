"""Download the raw English datasets for R1 into data/raw/.

What it does
    1. MultiFC: downloads ``multi_fc_publicdata.zip`` (authors' release, includes
       the Google snippets) and extracts train.tsv, dev.tsv, README.txt and the
       snippets/ folder. Other files in the zip (test.tsv with no labels, .pkl
       files) are not extracted; we never unpickle downloaded files.
    2. AVeriTeC: downloads the train (3,068) and dev (500) claims from the Hugging
       Face dataset ``pminervini/averitec`` through the datasets-server rows API, as
       plain JSON (no pyarrow needed). The official test labels are not public.

Inputs
    Internet access. No arguments needed.

Outputs
    data/raw/multifc/multi_fc_publicdata/{train.tsv, dev.tsv, README.txt, snippets/}
    data/raw/averitec/{train,dev}.json   (list of claim dicts, as on Hugging Face)

How to run (from the repo root; on Colab, run inside the Drive folder)
    python -m src.data.download_en
    python -m src.data.download_en --skip-multifc     # only AVeriTeC
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import zipfile

MULTIFC_URL = "https://www.dropbox.com/s/3v5oy3eddg3506j/multi_fc_publicdata.zip?dl=1"
AVERITEC_ROWS_API = ("https://datasets-server.huggingface.co/rows?dataset=pminervini/averitec"
                     "&config=default&split={split}&offset={offset}&length={length}")
AVERITEC_SPLITS = ("train", "dev")
PAGE_SIZE = 100  # the rows API returns at most 100 rows per call
USER_AGENT = "twincheck-data-download/1.0"


def fetch(url: str, retries: int = 3) -> bytes:
    """Download a URL and return its bytes, retrying a few times on errors.

    Args:
        url: URL to fetch.
        retries: number of attempts before giving up.

    Returns:
        Response body.
    """
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except OSError as err:
            if attempt == retries:
                raise
            print(f"  retry {attempt} after error: {err}")
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def download_multifc(raw_root: str) -> None:
    """Download and extract the MultiFC release (train, dev, README, snippets).

    Args:
        raw_root: ``data/raw`` folder.
    """
    folder = os.path.join(raw_root, "multifc")
    os.makedirs(folder, exist_ok=True)
    zip_path = os.path.join(folder, "multi_fc_publicdata.zip")
    if not os.path.exists(zip_path):
        print("Downloading MultiFC (~240 MB) ...")
        with open(zip_path, "wb") as f:
            f.write(fetch(MULTIFC_URL))
    if os.path.isdir(os.path.join(folder, "multi_fc_publicdata", "snippets")):
        print("MultiFC already extracted; skipping.")
        return
    wanted = ("multi_fc_publicdata/train.tsv", "multi_fc_publicdata/dev.tsv",
              "multi_fc_publicdata/README.txt")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if m in wanted or m.startswith("multi_fc_publicdata/snippets/")]
        print(f"Extracting {len(members)} MultiFC files ...")
        zf.extractall(folder, members=members)


def download_averitec(raw_root: str) -> None:
    """Download AVeriTeC train and dev as JSON files.

    Args:
        raw_root: ``data/raw`` folder.
    """
    folder = os.path.join(raw_root, "averitec")
    os.makedirs(folder, exist_ok=True)
    for split in AVERITEC_SPLITS:
        path = os.path.join(folder, f"{split}.json")
        if os.path.exists(path):
            print(f"AVeriTeC {split} already downloaded; skipping.")
            continue
        rows, offset, total = [], 0, None
        while total is None or offset < total:
            page = json.loads(fetch(AVERITEC_ROWS_API.format(
                split=split, offset=offset, length=PAGE_SIZE)))
            total = page["num_rows_total"]
            rows.extend(item["row"] for item in page["rows"])
            offset += PAGE_SIZE
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        print(f"AVeriTeC {split}: {len(rows)} claims -> {path}")


def main() -> None:
    """Parse arguments and download the requested datasets."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-root", default=os.path.join("data", "raw"))
    parser.add_argument("--skip-multifc", action="store_true")
    parser.add_argument("--skip-averitec", action="store_true")
    args = parser.parse_args()
    if not args.skip_multifc:
        download_multifc(args.raw_root)
    if not args.skip_averitec:
        download_averitec(args.raw_root)


if __name__ == "__main__":
    main()
