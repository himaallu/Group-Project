"""Unit tests for the F11 image card (no GPU, checkpoint or Drive needed).

Run: python -m pytest tests/demo/test_vision_card.py -q
The fake-model tests need torch and torchvision; they are skipped where those are missing.
"""

import io
import json

import numpy as np
import pytest
from PIL import Image

from src.demo import vision
from src.demo.app import ILLUSTRATIVE, run_check
from src.demo.vision_card import CAVEAT, IMAGE_NOTE, image_card, unavailable_card
from src.vision.infer import encode_jpeg, label_for, pixel_key, square96

POOL = [[[[16] * 64, [17] * 64], 2], [[[8] * 64, [9] * 64], 0]]   # two made-up JPEG settings


def picture(w=300, h=200, seed=0):
    """A random RGB test picture."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def test_label_at_threshold():
    assert label_for(0.5) == "AI-generated" and label_for(0.91) == "AI-generated"
    assert label_for(0.4999) == "Real" and label_for(0.0) == "Real"


def test_card_contents():
    ai = image_card({"p_ai": 0.913, "label": "AI-generated", "model": "ViT-B/16 (F5, seed 42)"}, picture())
    assert "AI-generated" in ai and "#cf222e" in ai and "P(AI) = 0.91" in ai
    assert 'aria-valuenow="91"' in ai and "data:image/jpeg;base64," in ai
    assert "ViT-B/16 (F5, seed 42)" in ai and "Show heatmap" not in ai
    real = image_card({"p_ai": 0.02, "label": "Real", "model": "m"}, picture(), heatmap=picture(50, 50))
    assert "Real" in real and "#1a7f37" in real and CAVEAT.replace("'", "&#x27;") in real
    assert "Show heatmap" in real
    assert "Image check not available" in unavailable_card("no checkpoint")


def test_square96_and_jpeg_follow_f3():
    a = square96(picture(300, 200))
    assert a.shape == (96, 96, 3) and a.dtype == np.uint8
    blob = encode_jpeg(a, "some-md5", POOL)
    assert blob == encode_jpeg(a, "some-md5", POOL)               # same key -> same bytes
    tables = Image.open(io.BytesIO(blob)).quantization
    assert [list(t) for t in tables.values()] in ([[16] * 64, [17] * 64], [[8] * 64, [9] * 64])
    assert pixel_key(a) == pixel_key(square96(picture(300, 200)))  # stable for the same picture


def test_text_only_is_unchanged(monkeypatch):
    monkeypatch.setenv("TWINCHECK_EXPLAINER", "template")
    monkeypatch.setenv("TWINCHECK_EXTRACTOR", "split")
    args = ("Dubai Metro suspended all lines today.", "The RTA said the lines ran normally.", "en",
            ILLUSTRATIVE)
    text_only = run_check(*args)
    assert text_only == run_check(*args, image=None)
    assert "Image check" not in text_only and IMAGE_NOTE not in text_only


def test_image_without_checkpoint_keeps_text(monkeypatch):
    monkeypatch.setenv("TWINCHECK_EXPLAINER", "template")
    monkeypatch.setenv("TWINCHECK_EXTRACTOR", "split")
    vision.configure(None, None)
    args = ("Dubai Metro suspended all lines today.", "The RTA said the lines ran normally.", "en")
    with_image = run_check(*args, image=picture())
    assert with_image.startswith(run_check(*args))                # claim cards identical
    assert "Image check not available" in with_image and IMAGE_NOTE in with_image
    assert vision.status() == "image check (F11): not available"


def test_image_examples_from_folder(tmp_path):
    picture().save(tmp_path / "a.jpg")
    (tmp_path / "examples.json").write_text(json.dumps([
        {"title": "AI image", "message": "m", "evidence": ["e"], "lang": "en", "image": "a.jpg"},
        {"title": "missing", "message": "m", "evidence": ["e"], "lang": "en", "image": "nope.jpg"}]))
    examples = vision.load_image_examples(str(tmp_path))
    assert [e["title"] for e in examples] == ["AI image"]
    assert vision.load_image_examples(str(tmp_path / "none")) == []


def test_check_image_with_fake_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.vision.infer import check_image

    class Fake(torch.nn.Module):
        """Returns a fixed logit for every input."""

        def __init__(self, logit):
            super().__init__()
            self.logit = torch.nn.Parameter(torch.tensor([logit]), requires_grad=False)

        def forward(self, x):
            return self.logit.expand(x.shape[0], 1)

    at = check_image(picture(), Fake(0.0), POOL)                    # sigmoid(0) = 0.5 -> AI
    assert at["label"] == "AI-generated" and abs(at["p_ai"] - 0.5) < 1e-6
    assert check_image(picture(), Fake(-2.0), POOL)["label"] == "Real"


def test_pool_from_cache_recovers_order(tmp_path):
    import hashlib
    import pickle

    from src.vision.infer import pool_from_cache

    true_pool = [[[[q] * 64, [q + 1] * 64], 2] for q in (5, 9, 13)]       # made-up pool of 3
    md5s = [hashlib.md5(str(i).encode()).hexdigest() for i in range(30)]
    a = square96(picture())
    cache = {"md5": md5s, "jpeg": [encode_jpeg(a, m, true_pool) for m in md5s], "jpeg_settings": 3}
    path = tmp_path / "cache.pkl"
    path.write_bytes(pickle.dumps(cache))
    assert pool_from_cache(path, tmp_path / "pool.json") == true_pool
    assert json.loads((tmp_path / "pool.json").read_text()) == true_pool


def test_lfs_pointer_gives_clear_message(tmp_path):
    pointer = tmp_path / "vit.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 343000000\n")
    vision.configure(str(pointer), "models/F11/jpeg_pool.json")
    assert "git lfs pull" in vision.image_html(picture())


def test_saved_pool_is_the_f3_size():
    from src.vision.infer import DEFAULT_POOL, load_jpeg_pool

    pool = load_jpeg_pool(DEFAULT_POOL)
    assert len(pool) == 500 and all(len(t) >= 2 and s in (0, 1, 2) for t, s in pool)


def test_summary_card_reports_both_without_changing_the_verdict(monkeypatch):
    from src.demo.vision_card import summary_card

    claims = [{"label": "unverified", "confidence": 1.0}]
    ai = summary_card(claims, {"p_ai": 0.91, "label": "AI-generated"})
    assert "Final result" in ai and "Unverified" in ai and "AI-generated" in ai and "caution" in ai
    assert "Real" in summary_card(claims, {"p_ai": 0.02, "label": "Real"})
    assert summary_card(claims, None) == ""                       # image check not available

    monkeypatch.setenv("TWINCHECK_EXPLAINER", "template")
    monkeypatch.setenv("TWINCHECK_EXTRACTOR", "split")
    monkeypatch.setattr(vision, "image_check", lambda img, **kw: ("<div>card</div>",
                                                            {"p_ai": 0.91, "label": "AI-generated"}))
    args = ("Dubai Metro suspended all lines today.", "The RTA said the lines ran normally.", "en")
    out = run_check(*args, image=picture())
    assert out.startswith(run_check(*args))                       # claim cards identical to text-only
    assert "Final result" in out and IMAGE_NOTE in out


def test_builtin_cases_cover_the_brief():
    from src.demo.image_examples import is_ai, load_examples

    cases = load_examples()
    assert len(cases) == 6
    truths = [c["truth"] for c in cases]
    assert truths[0] == "Real" and is_ai(truths[1]) and is_ai(truths[2]) and is_ai(truths[3])
    twin_a, twin_b = cases[4], cases[5]                          # image twin: same text, other image
    assert twin_a["message"] == twin_b["message"] and twin_a["evidence"] == twin_b["evidence"]
    assert twin_a["truth"] == "Real" and is_ai(twin_b["truth"]) and twin_b["blob"] is None


def test_truth_line_marks_right_and_wrong():
    hit = image_card({"p_ai": 0.99, "label": "AI-generated", "model": "m"}, picture(), truth="AI-generated (BigGAN)")
    miss = image_card({"p_ai": 0.01, "label": "Real", "model": "m"}, picture(), truth="AI-generated (SD v1.4)")
    real = image_card({"p_ai": 0.01, "label": "Real", "model": "m"}, picture(), truth="Real")
    assert "detector correct" in hit and "detector wrong" in miss and "detector correct" in real


def test_image_twin_keeps_the_claim_verdict(monkeypatch):
    from src.demo.image_examples import load_examples

    monkeypatch.setenv("TWINCHECK_EXPLAINER", "template")
    monkeypatch.setenv("TWINCHECK_EXTRACTOR", "split")
    scores = iter([{"p_ai": 0.01, "label": "Real"}, {"p_ai": 0.93, "label": "AI-generated"}])
    monkeypatch.setattr(vision, "image_check", lambda img, blob=None, truth=None: ("<div>card</div>", next(scores)))
    a, b = load_examples()[4:6]
    out_a = run_check(a["message"], "\n".join(a["evidence"]), a["lang"], image=a["image"], image_example=a)
    out_b = run_check(b["message"], "\n".join(b["evidence"]), b["lang"], image=b["image"], image_example=b)
    claims = run_check(a["message"], "\n".join(a["evidence"]), a["lang"])
    assert out_a.startswith(claims) and out_b.startswith(claims)  # identical claim cards for both twins


def test_image_only_shows_image_card(monkeypatch):
    vision.configure(None, None)
    out = run_check("", "", "auto", image=picture())
    assert "Image check not available" in out and "Paste a message" in out


def test_warning_when_library_missing_and_no_evidence(monkeypatch):
    from src.demo.app import library_warning

    monkeypatch.setattr("src.pipeline.component_status", lambda: {"get_evidence (R5)": "stub"})
    assert "Offline evidence library not loaded" in library_warning("")
    assert library_warning("The RTA said the lines ran normally.") == ""      # pasted evidence
    monkeypatch.setattr("src.pipeline.component_status", lambda: {"get_evidence (R5)": "real"})
    assert library_warning("") == ""
