"""AI-provenance tags in an image file, for the demo (F11).

What it does
    ``ai_provenance(raw)`` looks inside the file bytes for the labels that AI image tools
    write into their output: the IPTC digital-source type "trainedAlgorithmicMedia" (the
    standard "made by a generative model" value, used by Google, Adobe, Microsoft and
    others), C2PA content-credential manifests that name an AI generator, and a few
    known generator names in the metadata. A tag is strong evidence that the image is AI
    made; no tag proves nothing, because screenshots, re-saving and most social media strip
    metadata. Demo only; not used in any reported result.

Inputs
    raw (bytes): the uploaded file exactly as it was sent.

Outputs
    a short description of the tag found, or None.

How to run
    Used by src/demo/vision.py; tests in tests/demo/test_vision_card.py.
"""

from __future__ import annotations

# (bytes to look for, what to report); checked in this order
TAGS = [
    (b"trainedAlgorithmicMedia", "IPTC tag says made by a generative AI model"),
    (b"compositeWithTrainedAlgorithmicMedia", "IPTC tag says edited with generative AI"),
    (b"Made with Google AI", "Made with Google AI named in the file"),
    (b"Google Gemini", "Google Gemini named in the file"),
    (b"DALL-E", "OpenAI DALL-E named in the file"),
    (b"DALL\xc2\xb7E", "OpenAI DALL-E named in the file"),
    (b"OpenAI", "OpenAI named in the file"),
    (b"Adobe Firefly", "Adobe Firefly named in the file"),
    (b"Midjourney", "Midjourney named in the file"),
    (b"Stable Diffusion", "Stable Diffusion named in the file"),
]


def ai_provenance(raw: bytes | None) -> str | None:
    """The first AI tag found in the file bytes, or None."""
    if not raw:
        return None
    for marker, description in TAGS:
        if marker in raw:
            return description
    if b"c2pa" in raw and b"trainedAlgorithmic" in raw:
        return "C2PA content credentials: AI generated"
    return None
