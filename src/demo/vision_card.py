"""HTML card for the image check in the TwinCheck demo (F11).

What it does
    ``image_card(result, image)`` shows the result of ``src.vision.infer.check_image``
    next to the claim cards: a thumbnail, a red "AI-generated" or green "Real" badge,
    "P(AI) = 0.91" with a bar, the model name, and a caveat on every card. The detector
    was trained on BigGAN and misses most images from new generators, so "Real" only
    means no AI signal was found. The card is separate from the claim verdict, which
    never uses the image score (CLAUDE.md §3); ``IMAGE_NOTE`` says so under the results.
    Like the claim cards, it takes its text colour from the Gradio theme and uses only
    badge colours that read on light and dark backgrounds.

Inputs
    result: {"p_ai": float, "label": "AI-generated" | "Real", "model": str}; image: PIL image.

Outputs
    HTML strings (image_card, unavailable_card, IMAGE_NOTE).

How to run
    Used by src/demo/app.py; tests in tests/demo/test_vision_card.py.
"""

from __future__ import annotations

import base64
import html
import io

from PIL import Image

BADGE = {"AI-generated": "#cf222e", "Real": "#1a7f37"}      # same red / green as the claim cards
CAVEAT = ("Trained on BigGAN; on new generators it misses most AI images (recall 0.13 on "
          "Stable Diffusion v1.4), so 'Real' means no AI signal was found.")
IMAGE_NOTE = ("The claim verdict uses the text evidence only; the image result is shown "
              "separately and does not change it.")
CARD = "border:1px solid #d0d7de;border-radius:8px;padding:12px;margin:8px 0"   # as the claim cards
MUTED = "font-size:0.85em;opacity:0.75"


def thumbnail(image: Image.Image, max_side: int = 240) -> str:
    """The image as a small inline JPEG (data URI), so nothing is written to disk."""
    small = image.convert("RGB")
    small.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def image_card(result: dict, image: Image.Image, heatmap: Image.Image | None = None) -> str:
    """HTML card for one image check; ``heatmap`` (Grad-CAM) adds a "Show heatmap" toggle."""
    p_ai = float(result["p_ai"])
    label = result["label"]
    colour = BADGE.get(label, "#57606a")
    pct = round(100 * p_ai)
    heat = ""
    if heatmap is not None:
        heat = ('<details style="margin-top:6px"><summary style="cursor:pointer">Show heatmap '
                '(Grad-CAM: where the detector looked)</summary>'
                f'<img src="{thumbnail(heatmap)}" alt="Grad-CAM heat map" '
                'style="max-width:240px;border-radius:6px;margin-top:6px"></details>')
    return (
        f'<div style="{CARD}">'
        '<div style="font-weight:600;margin-bottom:6px">Image check (separate from the claim verdict)</div>'
        '<div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap">'
        f'<img src="{thumbnail(image)}" alt="uploaded image" '
        'style="max-width:160px;max-height:160px;border-radius:6px">'
        '<div style="flex:1;min-width:180px">'
        f'<div style="margin:2px 0 6px"><span style="background:{colour};color:#fff;'
        f'padding:2px 8px;border-radius:12px">{html.escape(label)}</span> '
        f'P(AI) = {p_ai:.2f}</div>'
        f'<div role="progressbar" aria-valuenow="{pct}" aria-valuemin="0" aria-valuemax="100" '
        f'aria-label="P(AI) {pct}%" style="background:#eaeef2;border-radius:4px;height:8px">'
        f'<div style="background:{colour};width:{pct}%;height:8px;border-radius:4px"></div></div>'
        f'<div style="{MUTED};margin-top:6px">Model: '
        f'{html.escape(result.get("model", ""))} · threshold 0.5</div>'
        f'<div style="{MUTED};margin-top:4px">{html.escape(CAVEAT)}</div>'
        f'{heat}</div></div></div>')


VERDICT_WORDS = {"supported": "Supported", "contradicted": "Contradicted", "unverified": "Unverified"}
READING = {   # (claim verdict, image label) -> what the two results mean together
    ("supported", "AI-generated"): "The evidence supports the claim, but the picture shows signs of AI "
                                   "generation: the claim may be true while this image is not an authentic photo.",
    ("contradicted", "AI-generated"): "The evidence contradicts the claim, and the picture shows signs of AI "
                                      "generation: treat this post as likely misleading.",
    ("unverified", "AI-generated"): "The evidence does not confirm the claim, and the picture shows signs of AI "
                                    "generation: treat this post with caution.",
    ("supported", "Real"): "The evidence supports the claim, and no AI signal was found in the picture.",
    ("contradicted", "Real"): "The evidence contradicts the claim. No AI signal was found in the picture, "
                              "but a real photo can still be used out of context.",
    ("unverified", "Real"): "The evidence does not confirm the claim, and no AI signal was found in the picture.",
}


def summary_card(claim_results: list[dict], image_result: dict | None) -> str:
    """Final result: each claim verdict and the image result together, in plain words.

    The claim verdicts are shown exactly as ``check`` gave them; the image result is
    never used to change them (CLAUDE.md §3). The text only explains what the two
    separate results mean together.
    """
    if image_result is None or not claim_results:
        return ""
    label, p_ai = image_result["label"], float(image_result["p_ai"])
    rows = []
    for r in claim_results:
        verdict = VERDICT_WORDS.get(r["label"], r["label"])
        reading = READING.get((r["label"], label), "")
        rows.append(f'<li><b>Claim:</b> {verdict} ({round(100 * float(r["confidence"]))}%, from the text '
                    f'evidence) · <b>Image:</b> {html.escape(label)} (P(AI) = {p_ai:.2f})<br>'
                    f'{html.escape(reading)}</li>')
    return (f'<div style="{CARD};border-width:2px">'
            '<div style="font-weight:600;margin-bottom:4px">Final result: text and image together</div>'
            f'<ul style="margin:0;padding-left:18px">{"".join(rows)}</ul>'
            f'<div style="{MUTED};margin-top:6px">The claim verdict is not changed by the image; '
            'the two results are reported side by side.</div></div>')


def unavailable_card(reason: str) -> str:
    """Shown instead of the image card when the detector cannot be loaded."""
    return (f'<div style="{CARD}"><b>Image check not available.</b> '
            f'<span style="opacity:0.75">{html.escape(reason)}</span></div>')


def image_note() -> str:
    """The line under the results saying the verdict ignores the image."""
    return f'<p style="font-size:0.85em;opacity:0.8;margin-top:4px">{html.escape(IMAGE_NOTE)}</p>'
