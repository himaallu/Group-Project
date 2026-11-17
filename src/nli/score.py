"""Score (claim, snippet) pairs with the fine-tuned TwinCheck NLI model (R4).

What it does
    ``score_pairs(claims, snippets)`` returns entail / contradict / neutral
    probabilities for each pair. ``src/pipeline.py`` (H4) calls it for its NLI step,
    and R5 uses it for explanation faithfulness. The model is loaded once and kept.
    It reuses Himasri's ``src/nli/score_zeroshot.py`` (same input order and columns),
    so fine-tuned and zero-shot scores can be swapped.

    Which model: the ``TWINCHECK_NLI`` environment variable if set; otherwise
    ``models/nli_v1`` if it holds weights; otherwise the copy on Hugging Face Hub
    (``RathiAbeth/twincheck-nli-v1``; the weights are too big for the repo's LFS quota).

    Importing this module needs torch; without it the import fails on purpose (see
    the import below), and the pipeline falls back to its word-overlap stand-in.

Inputs
    claims, snippets: two lists of strings of the same length.

Outputs
    [{"p_entail", "p_contradict", "p_neutral"}, ...], one dict per pair.

How to run
    python -m src.nli.score "Dubai Metro closed today" "The Dubai Metro ran normally today."
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

# Without torch this import fails, so src/pipeline.py (which imports this module only
# if it can) keeps its simple stand-in instead of crashing on a laptop with no torch.
import torch  # noqa: F401

LOCAL_MODEL = Path("models/nli_v1")
HUB_MODEL = "RathiAbeth/twincheck-nli-v1"


def model_name() -> str:
    """The NLI model to use (see the module docstring for the order)."""
    if os.environ.get("TWINCHECK_NLI"):
        return os.environ["TWINCHECK_NLI"]
    if any(LOCAL_MODEL.glob("*.safetensors")):
        return str(LOCAL_MODEL)
    return HUB_MODEL


@functools.lru_cache(maxsize=2)
def _load(name: str):
    """Load (tokenizer, model, device) once per model name."""
    from src.nli.score_zeroshot import load_model

    return load_model(name)


def score_pairs(claims: list[str], snippets: list[str], model: str | None = None,
                batch_size: int = 32) -> list[dict]:
    """NLI probabilities for aligned claims and snippets (premise = snippet).

    Args:
        claims: hypotheses.
        snippets: premises, same length as ``claims``.
        model: model name or folder; default ``model_name()``.

    Returns:
        One {"p_entail", "p_contradict", "p_neutral"} dict per pair.
    """
    if len(claims) != len(snippets):
        raise ValueError("claims and snippets must have the same length")
    if not claims:
        return []
    from src.nli.score_zeroshot import score_pairs as score_batch

    tok, nli, device = _load(model or model_name())
    table = score_batch(list(snippets), list(claims), tok, nli, device, batch_size)
    return table.round(4).to_dict(orient="records")


def main() -> None:
    """CLI: score one claim against one or more snippets."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("claim")
    ap.add_argument("snippets", nargs="+")
    ap.add_argument("--model")
    args = ap.parse_args()
    scores = score_pairs([args.claim] * len(args.snippets), args.snippets, args.model)
    for snippet, s in zip(args.snippets, scores):
        print(json.dumps({"snippet": snippet[:80], **s}, ensure_ascii=False))


if __name__ == "__main__":
    main()
