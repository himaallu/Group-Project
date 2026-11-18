"""Verdict for one claim at run time: the trained head behind ``src/pipeline.py`` (H4).

What it does
    ``predict_one(claim, snippets, nli)`` builds the same 10 features the head was
    trained on (``features.record_features``) and returns probabilities over
    LABELS (supported, contradicted, unverified):
      * relevance: the snippet's own ``relevance`` if every snippet has one (the
        offline index can return it), else Rathi's fine-tuned reranker (R2);
      * entities: spaCy on the claim (R2's ``src/data/entities.py``);
      * NLI: the scores the pipeline already computed.
    Heavy models are loaded once, on first use. If the reranker or spaCy cannot be
    loaded (no network, package missing) it prints one warning and continues without
    it (relevance 1.0 / no entities), so the demo keeps working.
    Entities come from an English spaCy model, so for Arabic and mixed claims they are
    skipped (entity features = 1.0, "nothing to contradict"). TODO: multilingual match.

Inputs
    claim     str
    snippets  [{"text", "url", "domain", optional "relevance"}, ...]
    nli       [{"p_entail", "p_contradict", "p_neutral"}, ...], one per snippet

Outputs
    numpy array of 3 probabilities (sums to 1). The pipeline then applies the
    calibration in ``models/verdict_calib.json``.

Settings (environment variables)
    TWINCHECK_VERDICT   model folder (default models/verdict_twin_s42, the H4 choice)
    TWINCHECK_RERANKER  reranker (default RathiAbeth/twincheck-reranker-v1)
    TWINCHECK_SPACY     spaCy model (default en_core_web_trf, then en_core_web_sm)

How to run
    from src.verdict.infer import predict_one
    predict_one("Dubai Metro suspended all lines", [{"text": "Green Line runs normally."}],
                [{"p_entail": 0.05, "p_contradict": 0.7, "p_neutral": 0.25}])
"""

from __future__ import annotations

import functools
import os
import re

import numpy as np
import pandas as pd
import torch

from src.verdict.features import FEATURES, record_features
from src.verdict.model import load_head, predict

DEFAULT_VERDICT = "models/verdict_twin_s42"
DEFAULT_RERANKER = "RathiAbeth/twincheck-reranker-v1"
SPACY_MODELS = ("en_core_web_trf", "en_core_web_sm")
ARABIC_CHAR = re.compile(r"[؀-ۿ]")


@functools.lru_cache(maxsize=1)
def _head():
    """(model, config) of the verdict head, loaded once."""
    return load_head(os.environ.get("TWINCHECK_VERDICT", DEFAULT_VERDICT))


@functools.lru_cache(maxsize=1)
def _reranker():
    """Rathi's reranker, or None (with one warning) if it cannot be loaded."""
    try:
        from src.retrieval.rerank import Reranker
        return Reranker(os.environ.get("TWINCHECK_RERANKER", DEFAULT_RERANKER))
    except Exception as err:  # noqa: BLE001 - any load failure means "run without it"
        print(f"warning: reranker not available ({type(err).__name__}: {err}); "
              f"using relevance 1.0 for every snippet")
        return None


@functools.lru_cache(maxsize=1)
def _spacy():
    """spaCy NER pipeline, or None (with one warning) if no model is installed."""
    names = [os.environ["TWINCHECK_SPACY"]] if os.environ.get("TWINCHECK_SPACY") else SPACY_MODELS
    for name in names:
        try:
            from src.data.entities import load_nlp
            return load_nlp(name)
        except Exception:  # noqa: BLE001 - try the next model
            continue
    print("warning: no spaCy model found; entity-match features set to 1.0")
    return None


def relevance_rows(claim: str, snippets: list[dict]) -> dict[int, tuple[float]] | None:
    """{snippet_idx: (relevance,)} from the snippets or the reranker; None if unavailable."""
    if snippets and all(s.get("relevance") is not None for s in snippets):
        return {i: (float(s["relevance"]),) for i, s in enumerate(snippets)}
    reranker = _reranker()
    if reranker is None:
        return None
    return {i: (r,) for i, r in enumerate(reranker.score(claim, [s["text"] for s in snippets]))}


def claim_entities(claim: str) -> list[dict]:
    """spaCy entities of an English claim; [] for Arabic text or without spaCy."""
    if ARABIC_CHAR.search(claim):
        return []
    nlp = _spacy()
    if nlp is None:
        return []
    from src.data.entities import doc_entities
    return doc_entities(nlp(claim))


def predict_one(claim: str, snippets: list[dict], nli: list[dict]) -> np.ndarray:
    """Probabilities over (supported, contradicted, unverified) for one claim.

    Args:
        claim: the claim text.
        snippets: evidence snippets (dicts with at least "text").
        nli: NLI probabilities per snippet, same order as ``snippets``.

    Returns:
        Array of 3 probabilities.
    """
    model, config = _head()
    rec = {"id": "pipeline", "claim": claim, "evidence": snippets}
    nli_rows = {i: (s["p_entail"], s["p_contradict"], s["p_neutral"]) for i, s in enumerate(nli)}
    rel = relevance_rows(claim, snippets) if snippets and not config.get("no_relevance") else None
    feats, _ = record_features(rec, nli_rows, rel, claim_entities(claim))
    table = pd.DataFrame([{f: feats[f] for f in FEATURES}])
    logits = predict(model, config, table)
    return torch.softmax(torch.from_numpy(logits[0]), dim=-1).numpy()
