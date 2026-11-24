"""Second-opinion AI-image detector from the Hugging Face Hub, for the demo only (F11).

What it does
    Our own detector (F5 ViT-B/16) learned BigGAN fingerprints and misses most images from
    newer generators (recall 0.13 on Stable Diffusion v1.4). For the demo, a public
    detector trained on newer generators is shown next to it as a clearly labelled second
    opinion. Default: ``Organika/sdxl-detector`` (image classification with an "artificial"
    and a "human" class). It was NOT trained or tested by us, so it is never used for a
    reported number; the card says so.
    ``external_p_ai(image, classifier)`` = the total score of the classes whose name means
    AI-made ("artificial", "fake", "ai", "generated", "synthetic").

Inputs
    a PIL image; the model id (--external-detector in src/demo/app.py; "none" turns it off).

Outputs
    P(AI) between 0 and 1.

How to run
    Used by src/demo/vision.py; tests in tests/demo/test_vision_card.py.
"""

from __future__ import annotations

DEFAULT_MODEL = "Organika/sdxl-detector"
AI_WORDS = ("artificial", "fake", "ai", "generated", "synthetic")


def is_ai_class(name: str) -> bool:
    """True if a class name means AI-made (e.g. 'artificial', 'FAKE', 'AI-generated')."""
    words = name.lower().replace("-", " ").replace("_", " ").split()
    return any(w in AI_WORDS for w in words)


def load_external(model_id: str = DEFAULT_MODEL):
    """A transformers image-classification pipeline (GPU if available)."""
    import torch
    from transformers import pipeline

    return pipeline("image-classification", model=model_id, device=0 if torch.cuda.is_available() else -1)


def external_p_ai(image, classifier) -> float:
    """P(AI) from the external detector: the summed score of its AI-made classes."""
    scores = classifier(image.convert("RGB"), top_k=None)
    return float(sum(s["score"] for s in scores if is_ai_class(s["label"])))
