"""TwinCheck end-to-end pipeline: ``check(message, evidence=None, lang=None)`` (H4).

What it does
    Message -> claims -> evidence -> NLI per snippet -> verdict -> calibration ->
    cited explanation, one result per claim. This is the single entry point used by
    the demo (Rathi R6) and the Arabic experiments (Youssef Y5).

    Each step is a small function. Until a teammate's part lands, a step uses a
    simple stand-in ("stub") that returns output of the right shape, marked
    ``TODO(<task-id>)``. When the real module exists it is used automatically, so
    code written against ``check()`` today keeps working.

        step              real module (owner)                 stand-in until then
        extract_claims    src/claims/extract.py (Youssef Y3)  split into sentences
        get_evidence      src/retrieval/index.py (Rathi R5)   evidence passed in, else none
        score_nli         models/nli_v1 (Rathi R4)            word-overlap scores
        verdict           src/verdict/model.py (Himasri H3)   rule on max entail/contradict
        calibrate         models/verdict_calib.json (H4)      no change if file missing
        explain           src/explain/generate.py (Rathi R5)  quote the best snippet

    ``component_status()`` says which steps are real and which are stand-ins.
    By design (CLAUDE.md §3) the verdict never sees any "is the image AI-made?" score.

Inputs
    message   free text (English, Arabic or mixed), may hold several claims.
    evidence  optional list of snippets (str or {"text", "url", "domain"}) used for
              every claim; if None the offline index is searched.
    lang      optional "en" / "ar" / "mixed"; detected from the text if None.

Outputs
    A list with one dict per claim:
        {"claim": str, "language": "en"|"ar"|"mixed",
         "label": "supported"|"contradicted"|"unverified",
         "confidence": float,                       # probability of ``label``
         "probs": {"supported", "contradicted", "unverified"},
         "evidence": [{"text", "url", "domain", "p_entail", "p_contradict", "p_neutral"}],
         "explanation": str}                        # cites snippets as [1]..[3]

How to run
    python -m src.pipeline "Dubai Metro suspended all lines today because of flooding."
    python -m src.pipeline --status
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import numpy as np

from src.eval.metrics import LABELS

CALIB_PATH = Path("models/verdict_calib.json")
TOP_K = 5               # snippets kept per claim
MAX_CITED = 3           # explanations cite at most [1]..[3]
ARABIC_CHAR = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
LATIN_CHAR = re.compile(r"[A-Za-z]")
WORD = re.compile(r"\w+", re.UNICODE)


def _optional(module: str, attr: str):
    """Return ``module.attr`` if a teammate's module exists, else None."""
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError):
        return None


# --------------------------------------------------------------------------
# Step 0: language
# --------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Guess "en", "ar" or "mixed" from the share of Arabic letters.

    >= 80% Arabic letters -> "ar", <= 20% -> "en", anything between -> "mixed".
    Text with no letters at all counts as "en".
    """
    n_ar = len(ARABIC_CHAR.findall(text))
    n_lat = len(LATIN_CHAR.findall(text))
    if n_ar + n_lat == 0:
        return "en"
    share = n_ar / (n_ar + n_lat)
    if share >= 0.8:
        return "ar"
    if share <= 0.2:
        return "en"
    return "mixed"


# --------------------------------------------------------------------------
# Step 1: claims
# --------------------------------------------------------------------------

def extract_claims(message: str, lang: str) -> list[str]:
    """Split a message into check-worthy claims.

    Uses Youssef's ``src.claims.extract.extract_claims(message) -> list[str]`` when
    it exists. TODO(Y3): until then, split on sentence ends (. ! ? ؟ and new lines)
    and keep pieces with at least 4 words.
    """
    real = _optional("src.claims.extract", "extract_claims")
    if real is not None:
        return list(real(message))
    pieces = re.split(r"(?<=[.!?؟])\s+|\n+", message)
    claims = [p.strip() for p in pieces if len(WORD.findall(p)) >= 4]
    return claims or ([message.strip()] if message.strip() else [])


# --------------------------------------------------------------------------
# Step 2: evidence
# --------------------------------------------------------------------------

def _as_snippet(item) -> dict:
    """Normalise a snippet given as a string or dict to {"text", "url", "domain"}."""
    if isinstance(item, str):
        return {"text": item, "url": None, "domain": None}
    return {"text": item["text"], "url": item.get("url"), "domain": item.get("domain")}


def get_evidence(claim: str, evidence: list | None, k: int = TOP_K) -> list[dict]:
    """Evidence snippets for one claim.

    If ``evidence`` is given it is used as is (first ``k``). Otherwise Rathi's
    offline index ``src.retrieval.index.search(claim, k) -> list[dict]`` is used.
    TODO(R5): until the index exists, no evidence -> the verdict is "unverified".
    """
    if evidence is not None:
        return [_as_snippet(e) for e in evidence][:k]
    search = _optional("src.retrieval.index", "search")
    if search is not None:
        return [_as_snippet(e) for e in search(claim, k)][:k]
    return []


# --------------------------------------------------------------------------
# Step 3: NLI per snippet
# --------------------------------------------------------------------------

def score_nli(claim: str, snippets: list[dict]) -> list[dict]:
    """Entail / contradict / neutral probabilities for each (snippet, claim) pair.

    Uses ``src.nli.score.score_pairs(claims, snippets) -> list[dict]`` (Rathi R4,
    fine-tuned mDeBERTa in ``models/nli_v1``) when it exists.
    TODO(R4): until then a crude stand-in: word overlap between claim and snippet
    becomes p_entail, the rest p_neutral, p_contradict = 0. It only keeps the
    output shape right; do not report numbers from it.
    """
    real = _optional("src.nli.score", "score_pairs")
    if real is not None:
        return list(real([claim] * len(snippets), [s["text"] for s in snippets]))
    claim_words = {w.lower() for w in WORD.findall(claim)}
    scores = []
    for s in snippets:
        snip_words = {w.lower() for w in WORD.findall(s["text"])}
        overlap = len(claim_words & snip_words) / max(len(claim_words), 1)
        p_entail = round(min(overlap, 1.0) * 0.9, 4)
        scores.append({"p_entail": p_entail, "p_contradict": 0.0,
                       "p_neutral": round(1.0 - p_entail, 4)})
    return scores


# --------------------------------------------------------------------------
# Step 4: verdict + Step 5: calibration
# --------------------------------------------------------------------------

def verdict_probs(claim: str, snippets: list[dict], nli: list[dict]) -> np.ndarray:
    """Probabilities over LABELS (supported, contradicted, unverified).

    Uses ``src.verdict.model.predict_one(claim, snippets, nli) -> array(3)``
    (Himasri H3) when it exists. TODO(H3): until then a simple rule: support =
    max p_entail, contradict = max p_contradict, unverified = what is left.
    No snippets -> certainly unverified.
    """
    real = _optional("src.verdict.model", "predict_one")
    if real is not None:
        return np.asarray(real(claim, snippets, nli), dtype=float)
    if not nli:
        return np.array([0.0, 0.0, 1.0])
    sup = max(s["p_entail"] for s in nli)
    con = max(s["p_contradict"] for s in nli)
    unv = max(1.0 - sup - con, 0.0)
    probs = np.array([sup, con, unv]) + 1e-6
    return probs / probs.sum()


def calibrate(probs: np.ndarray) -> tuple[str, np.ndarray]:
    """Apply temperature + Unverified threshold from ``models/verdict_calib.json``.

    If the settings file does not exist yet (before H4 fits it), the argmax of the
    raw probabilities is returned unchanged.
    """
    probs = np.asarray(probs, dtype=float)
    if not CALIB_PATH.exists():
        return LABELS[int(probs.argmax())], probs
    from src.verdict.calibrate import apply_calibration, load_settings
    s = load_settings(CALIB_PATH)
    # log-probabilities are valid logits: softmax(log p) == p, so T and tau apply as fitted.
    labels, cal = apply_calibration(np.log(probs + 1e-12)[None, :], s["T"], s["tau"])
    return labels[0], cal[0]


# --------------------------------------------------------------------------
# Step 6: explanation
# --------------------------------------------------------------------------

def explain(claim: str, label: str, snippets: list[dict], nli: list[dict], lang: str) -> str:
    """Short note citing the evidence as [1]..[3].

    Uses ``src.explain.generate.generate(claim, label, snippets, lang) -> str``
    (Rathi R5, Qwen3-8B) when it exists. TODO(R5): until then, quote the snippet that
    best supports the verdict.
    """
    real = _optional("src.explain.generate", "generate")
    cited = snippets[:MAX_CITED]
    if real is not None:
        return real(claim, label, cited, lang)
    if not cited:
        return "No relevant evidence was found for this claim."
    key = {"supported": "p_entail", "contradicted": "p_contradict"}.get(label, "p_neutral")
    best = max(range(len(cited)), key=lambda i: nli[i][key])
    quote = cited[best]["text"][:200]
    return f"Verdict: {label}. Key evidence [{best + 1}]: \"{quote}\""


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def check(message: str, evidence: list | None = None, lang: str | None = None) -> list[dict]:
    """Fact-check every claim in a message.

    Args:
        message: the text to check (English, Arabic or mixed).
        evidence: optional snippets (str or dict) to use for every claim instead of
            searching the offline index.
        lang: optional language override ("en", "ar", "mixed").

    Returns:
        One result dict per claim (fields listed in the module docstring). An empty
        message gives an empty list.
    """
    results = []
    for claim in extract_claims(message, lang or detect_language(message)):
        claim_lang = lang or detect_language(claim)
        snippets = get_evidence(claim, evidence)
        nli = score_nli(claim, snippets)
        label, probs = calibrate(verdict_probs(claim, snippets, nli))
        results.append({
            "claim": claim,
            "language": claim_lang,
            "label": label,
            "confidence": float(probs[LABELS.index(label)]),
            "probs": {lab: float(p) for lab, p in zip(LABELS, probs)},
            "evidence": [{**s, **n} for s, n in zip(snippets, nli)],
            "explanation": explain(claim, label, snippets, nli, claim_lang),
        })
    return results


def component_status() -> dict[str, str]:
    """Which steps use a teammate's real module ("real") and which a stand-in ("stub")."""
    found = {
        "extract_claims (Y3)": _optional("src.claims.extract", "extract_claims"),
        "get_evidence (R5)": _optional("src.retrieval.index", "search"),
        "score_nli (R4)": _optional("src.nli.score", "score_pairs"),
        "verdict (H3)": _optional("src.verdict.model", "predict_one"),
        "calibrate (H4)": CALIB_PATH if CALIB_PATH.exists() else None,
        "explain (R5)": _optional("src.explain.generate", "generate"),
    }
    return {name: "real" if obj is not None else "stub" for name, obj in found.items()}


def main() -> None:
    """CLI: check one message and print the results as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("message", nargs="?", help="message to check")
    ap.add_argument("--evidence", nargs="*", help="evidence snippets (optional)")
    ap.add_argument("--lang", choices=["en", "ar", "mixed"])
    ap.add_argument("--status", action="store_true", help="show which steps are stubs")
    args = ap.parse_args()
    if args.status or not args.message:
        print(json.dumps(component_status(), indent=2))
        return
    print(json.dumps(check(args.message, args.evidence, args.lang), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
