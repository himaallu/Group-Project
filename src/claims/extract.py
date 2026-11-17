"""Claim extraction: split a messy message into check-worthy claims (task Y3, part 2).

What it does
    extract_claims(message) -> list[str]
    Given a forwarded-style message (English, Arabic or mixed), returns the factual
    claims in it, each as one standalone sentence in its ORIGINAL language, without
    chat text ("please share", "is this true?", greetings, emojis).

    Two backends:
      - "qwen" (default when a GPU is available): Qwen3-8B, 4-bit, few-shot with 6
        examples from data/claims/gold*.jsonl (2 English, 2 Arabic, 2 mixed), greedy
        decoding, thinking off; the model answers with a JSON list of strings.
      - "split" (no GPU, or TWINCHECK_EXTRACTOR=split): a rule-based fallback that
        splits on sentence ends / new lines and drops chat phrases. It keeps
        pipeline.py and its tests fast on CPU.
    Set TWINCHECK_EXTRACTOR=qwen|split to force a backend and
    TWINCHECK_EXTRACT_MODEL to use another model (e.g. Qwen/Qwen3-4B).

Inputs
    a message string; few-shot examples from data/claims/gold.jsonl (or the draft file)

Outputs
    list of claim strings

How to run
    python -m src.claims.extract "BREAKING 🚨 Dubai Metro closed all lines today. Please share"
    from src.claims.extract import extract_claims
"""

from __future__ import annotations

import json
import os
import re
import sys

MODEL = os.environ.get("TWINCHECK_EXTRACT_MODEL", "Qwen/Qwen3-8B")
GOLD_FILES = ["data/claims/gold.jsonl", "data/claims/gold_draft_0001.jsonl"]
# Fixed few-shot examples (2 per language); evaluate.py leaves these out.
FEWSHOT_IDS = ["claims_0001", "claims_0004", "claims_0005", "claims_0008", "claims_0006", "claims_0012"]
WORD = re.compile(r"\w+", re.UNICODE)

INSTRUCTION = (
    "You extract factual claims from forwarded social-media messages.\n"
    "Return ONLY a JSON list of strings. Each string is one checkable factual claim, "
    "written as a standalone sentence in the SAME language it appears in the message "
    "(do not translate). Remove chat text such as greetings, 'please share', 'is this "
    "true?', 'they are saying', emojis and opinions. If there is no factual claim, return []."
)

# Chat phrases the rule-based fallback removes (English and Arabic).
CHAT = re.compile(
    r"(forwarded as received:?|guys did you see this\?*|breaking|please share|share before.*|"
    r"is this true\??|can someone confirm\??|forward to all groups|stay safe everyone|"
    r"apparently:?|they're saying:?|news says:?|also:?|plus:?|another thing:?|on top of that:?|"
    r"منقول كما وصلني:?|يا جماعة شفتوا هذا[؟?]?|عاجل|انشروها|انشروا قبل الحذف!*|هل هذا صحيح[؟?]?|"
    r"أرسلوها لكل القروبات|الله يستر|يقولون إن|الخبر يقول إن|انتشر أن|وأيضاً|وكمان يقولون|وفوق هذا|وشي ثاني:?)",
    re.IGNORECASE)

_model = None  # (tokenizer, model), loaded once


def backend() -> str:
    """Pick 'qwen' when a GPU is available (or forced by TWINCHECK_EXTRACTOR), else 'split'."""
    forced = os.environ.get("TWINCHECK_EXTRACTOR")
    if forced in ("qwen", "split"):
        return forced
    try:
        import torch
        return "qwen" if torch.cuda.is_available() else "split"
    except ImportError:
        return "split"


def split_claims(message: str) -> list[str]:
    """Rule-based fallback: sentence/line split, chat phrases removed, >= 4 words kept."""
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", " ", message)  # emojis
    pieces = re.split(r"(?<=[.!?؟])\s+|\n+", text)
    claims = []
    for p in pieces:
        p = CHAT.sub(" ", p).strip(" :!-")
        p = re.sub(r"\s+", " ", p).strip()
        if len(WORD.findall(p)) >= 4:
            claims.append(p)
    return claims


def load_fewshot() -> list[dict]:
    """Return the 6 few-shot examples (message + gold claims) from the first gold file found."""
    for path in GOLD_FILES:
        if os.path.exists(path):
            rows = {r["id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}
            shots = [rows[i] for i in FEWSHOT_IDS if i in rows]
            if shots:
                return shots
    return []


def build_messages(message: str, shots: list[dict]) -> list[dict]:
    """Chat messages for the model: instruction, few-shot pairs, then the new message."""
    chat = [{"role": "system", "content": INSTRUCTION}]
    for s in shots:
        chat.append({"role": "user", "content": s["message"]})
        chat.append({"role": "assistant", "content": json.dumps(s["claims"], ensure_ascii=False)})
    chat.append({"role": "user", "content": message})
    return chat


def parse_json_list(text: str) -> list[str] | None:
    """Read the first JSON list of strings in the model output (None if there is none)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return None
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return [str(x).strip() for x in items if str(x).strip()] if isinstance(items, list) else None


def _load():
    """Load the LLM once (4-bit on GPU)."""
    global _model
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16))
        _model = (tok, model)
    return _model


def extract_batch(messages: list[str], max_new_tokens: int = 384) -> list[list[str]]:
    """Extract claims for several messages at once (faster on GPU)."""
    if backend() == "split":
        return [split_claims(m) for m in messages]
    import torch

    tok, model = _load()
    shots = load_fewshot()
    prompts = [tok.apply_chat_template(build_messages(m, shots), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for m in messages]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    # If the model's answer is not a JSON list, fall back to the rule-based split.
    return [parse_json_list(t) if parse_json_list(t) is not None else split_claims(m)
            for t, m in zip(texts, messages)]


def extract_claims(message: str) -> list[str]:
    """Return the factual claims in one message (interface used by src/pipeline.py)."""
    return extract_batch([message])[0]


if __name__ == "__main__":
    print(json.dumps(extract_claims(" ".join(sys.argv[1:])), ensure_ascii=False, indent=1))
