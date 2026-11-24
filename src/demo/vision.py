"""Image check for the TwinCheck demo: settings, lazy model loading, examples (F11).

What it does
    ``configure(ckpt, jpeg_pool, examples_dir)`` stores the paths given on the command line
    (``python -m src.demo.app --vision-ckpt ... --jpeg-pool ...``). ``image_html(image)``
    loads the F5 ViT-B/16 detector once, on first use, scores the picture with
    ``src.vision.infer.check_image`` and returns the image card (with a Grad-CAM toggle).
    If the checkpoint or the pool is missing, or loading fails, it returns
    "Image check not available" and the text demo keeps working.
    ``load_image_examples(dir)`` reads the image example buttons from
    ``<dir>/examples.json``: a list of {"title", "message", "evidence": [...], "lang",
    "image"} with the image file next to it. The images stay on Drive, not in git.
    Nothing here is passed to ``src.pipeline.check``: the claim verdict never sees the
    image score (CLAUDE.md §3).

Inputs
    Paths from the command line; a PIL image from the browser.

Outputs
    HTML strings; a status string for the footer.

How to run
    Used by src/demo/app.py; tests in tests/demo/test_vision_card.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.demo.vision_card import image_card, unavailable_card

SETTINGS = {"ckpt": None, "jpeg_pool": None, "examples_dir": None}
_state = {"model": None, "pool": None, "error": None}


def configure(ckpt: str | None, jpeg_pool: str | None, examples_dir: str | None = None) -> None:
    """Remember the paths from the command line; nothing is loaded yet."""
    SETTINGS.update(ckpt=ckpt, jpeg_pool=jpeg_pool, examples_dir=examples_dir)
    _state.update(model=None, pool=None, error=None)


def _load() -> tuple:
    """(model, pool), loaded once; raises RuntimeError with a short reason if not possible."""
    if _state["model"] is not None:
        return _state["model"], _state["pool"]
    if _state["error"]:
        raise RuntimeError(_state["error"])
    ckpt, pool_path = SETTINGS["ckpt"], SETTINGS["jpeg_pool"]
    if not ckpt or not Path(ckpt).exists():
        _state["error"] = "Detector checkpoint not found (start the app with --vision-ckpt)."
    elif Path(ckpt).stat().st_size < 1_000_000:       # a Git LFS pointer, not the weights
        _state["error"] = "Detector checkpoint is a Git LFS pointer: run `git lfs pull` first."
    elif not pool_path or not Path(pool_path).exists():
        _state["error"] = "JPEG settings file not found (start the app with --jpeg-pool)."
    else:
        try:
            from src.vision.infer import load_detector, load_jpeg_pool

            _state["model"], _state["pool"] = load_detector(ckpt), load_jpeg_pool(pool_path)
            return _state["model"], _state["pool"]
        except Exception as err:                       # keep the text demo running
            _state["error"] = f"Detector could not be loaded ({type(err).__name__}: {err})."
    raise RuntimeError(_state["error"])


def image_check(image) -> tuple[str, dict | None]:
    """(image card HTML, result) for ``image`` (PIL); result is None if the check is not available."""
    try:
        model, pool = _load()
    except RuntimeError as err:
        return unavailable_card(str(err)), None
    from src.vision.infer import check_image, gradcam_overlay

    result = check_image(image, model, pool)
    try:
        heatmap = gradcam_overlay(image, model, pool)
    except Exception:                                  # the heat map is optional
        heatmap = None
    return image_card(result, image, heatmap), result


def image_html(image) -> str:
    """Image card for ``image`` (PIL), or the "not available" card."""
    return image_check(image)[0]


def status() -> str:
    """Short status for the footer components line."""
    if _state["model"] is not None:
        return "image check (F11): real"
    if _state["error"]:
        return "image check (F11): not available"
    return "image check (F11): loads on first image" if SETTINGS["ckpt"] else "image check (F11): off"


def load_image_examples(examples_dir: str | None = None) -> list[dict]:
    """Image example buttons from ``<dir>/examples.json``; [] if the folder or a file is missing."""
    folder = examples_dir or SETTINGS["examples_dir"]
    if not folder or not (Path(folder) / "examples.json").exists():
        return []
    folder = Path(folder)
    index = folder / "examples.json"
    examples = []
    for ex in json.loads(index.read_text(encoding="utf-8")):
        path = folder / ex["image"]
        if path.exists():
            examples.append({**ex, "image_path": str(path)})
    return examples
