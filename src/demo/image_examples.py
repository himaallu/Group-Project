"""Text + image example cases for the TwinCheck demo, built from files in the repo (F11).

What it does
    Six buttons, one per case in the Team Brief (§2–§5: image side, image twins, the rule
    that the verdict never uses the authenticity score). Every image has a known label:
      1. real photo + true claim                      -> Supported, image "Real"
      2. AI image (BigGAN) + true caption             -> still Supported, image "AI-generated"
      3. AI image (Stable Diffusion, detected) + false claim -> Contradicted, "AI-generated"
      4. AI image the detector misses (Stable Diffusion) + claim the evidence does not
         settle -> Unverified, image "Real" although it is AI: the known limit
      5a/5b. image twin: a real photo and its SDXL img2img edit, same caption and
         evidence -> the claim verdict must be identical; only the image card may change
    Images 1–4 and 5a come from the F3 96 px cache (scored on exactly the input F10 used,
    so the scores reproduce results/F10/preds_vit_b16_s42_check.csv); 5b is the SDXL edit
    from data/copies/vision/genimage_sdxl_img2img_500/. They were picked with a fixed rule
    (seed 42) from the F10 predictions, not by hand: the first sample of each group
    (test real with P < 0.5; validation BigGAN with P >= 0.5; test SD with P >= 0.5; test SD
    with P < 0.5; test real images with an SDXL edit). Captions describe what each picture
    shows; claims and evidence are made up ("Illustrative example").
    Images: GenImage, CC BY-NC-SA 4.0 (non-commercial, with credit).

Inputs
    data/processed/vision/genimage_debiased96_jpeg.pkl, the SDXL edit JPEG.

Outputs
    ``load_examples()`` -> list of {"title", "case", "message", "evidence", "lang",
    "truth", "image" (PIL), "blob" (cached F3 JPEG bytes or None)}.

How to run
    Used by src/demo/app.py; tests in tests/demo/test_vision_card.py.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

CACHE = Path("data/processed/vision/genimage_debiased96_jpeg.pkl")
SDXL_DIR = Path("data/copies/vision/genimage_sdxl_img2img_500/images")
CREDIT = "Example images: GenImage (CC BY-NC-SA 4.0); case 5b is an SDXL img2img edit (F2)."

_BRIDGE_CLAIM = "The steel bridge reopened to traffic on Monday after repair work."
_BRIDGE_EVIDENCE = ["The transport authority said the steel bridge reopened to traffic on Monday "
                    "once the repair work was finished.",
                    "Cars crossed the repaired bridge again on Monday morning."]

CASES = [
    {"title": "1 · Real photo, true claim", "md5": "fbdfacd7f03b78b526f7c5b31df6a625",
     "truth": "Real", "case": "Real photo with a claim the evidence supports.",
     "message": "A batter trained in an indoor batting cage at the club's new practice centre.",
     "evidence": ["The club opened its new practice centre with indoor batting cages this month.",
                  "Players trained in the indoor batting cages at the new centre on Saturday."],
     "lang": "en"},
    {"title": "2 · AI image, true caption", "md5": "3e361101db9ed5080fe397a2ea325bb8",
     "truth": "AI-generated (BigGAN)",
     "case": "AI-generated picture with a TRUE caption: the image card says AI-generated, "
             "but the claim verdict still follows the evidence.",
     "message": "Green apples are sold at Dubai's Al Aweer fruit and vegetable market.",
     "evidence": ["Traders at the Al Aweer Central Fruit and Vegetable Market in Dubai sell green "
                  "apples imported from several countries.",
                  "Green apples are among the fruits on sale at the Al Aweer market."],
     "lang": "en"},
    {"title": "3 · AI image, false claim", "md5": "a78c8351c73a31521f3e899c7029781a",
     "truth": "AI-generated (Stable Diffusion v1.4)",
     "case": "AI-generated picture with a claim the evidence contradicts.",
     "message": "All dogs were banned from every public park in Dubai this week.",
     "evidence": ["Dubai Municipality said dogs on a leash remain allowed in designated public parks.",
                  "No new ban on dogs in public parks was announced this week, officials said."],
     "lang": "en"},
    {"title": "4 · AI image the detector misses", "md5": "124f03822c8f92c0096cddc633809865",
     "truth": "AI-generated (Stable Diffusion v1.4)",
     "case": "AI picture from a generator the detector never saw: it is likely called 'Real' "
             "(the known limit: recall 0.13 on Stable Diffusion v1.4).",
     "message": "A new species of striped gecko was discovered in the Hajar Mountains.",
     "evidence": ["The Hajar Mountains are a popular hiking destination in the winter months.",
                  "A wildlife photography exhibition opened in Sharjah this week."],
     "lang": "en"},
    {"title": "5a · Image twin: original", "md5": "d9d8d52be0b419e4de05ed93db5f6e61",
     "truth": "Real", "case": "Image twin, original photo. Compare with 5b: same caption and "
                              "evidence, so the claim verdict must not change.",
     "message": _BRIDGE_CLAIM, "evidence": _BRIDGE_EVIDENCE, "lang": "en"},
    {"title": "5b · Image twin: SDXL-edited", "file": "865ee332a175307e.jpg",
     "truth": "AI-edited (SDXL img2img of 5a)",
     "case": "Image twin, the same photo edited by SDXL (AI fingerprints, same content). "
             "The claim verdict must be identical to 5a; only the image card may change.",
     "message": _BRIDGE_CLAIM, "evidence": _BRIDGE_EVIDENCE, "lang": "en"},
]

_cache: dict = {}


def _blobs() -> dict:
    """md5 -> cached F3 JPEG bytes, for the example images only (loaded once)."""
    if not _cache:
        import pickle

        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
        wanted = {c["md5"] for c in CASES if "md5" in c}
        _cache.update({m: b for m, b in zip(cache["md5"], cache["jpeg"]) if m in wanted})
    return _cache


def load_examples() -> list[dict]:
    """The six cases with their images; [] if the cache or an image file is missing."""
    if not CACHE.exists() or CACHE.stat().st_size < 1_000_000:
        return []
    examples = []
    for case in CASES:
        if "md5" in case:
            blob = _blobs().get(case["md5"])
            if blob is None:
                continue
            image = Image.open(io.BytesIO(blob)).convert("RGB")
        else:
            path = SDXL_DIR / case["file"]
            if not path.exists():
                continue
            blob, image = None, Image.open(path).convert("RGB")
        examples.append({**case, "image": image, "blob": blob})
    return examples


def is_ai(truth: str) -> bool:
    """True for a known label that means AI-made or AI-edited."""
    return truth.startswith("AI")
