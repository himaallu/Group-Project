"""Paraphrase twins ``_P`` with Qwen3-8B (R3, CLAUDE.md §2 and §4).

What it does
    Asks Qwen3-8B (4-bit, thinking switched off) to reword each source claim with
    the same meaning and the same names, dates and numbers. A paraphrase is kept
    only if
        * spaCy finds the same entity texts in it as in the claim, and
        * it has the same numbers (digit strings), and
        * it is really different from the claim (not just case or punctuation).
    The first attempt is greedy; up to 2 retries use sampling. Claims that never
    pass are dropped and counted. Evidence and label stay the same as the source.

Inputs
    Source records, a ``Paraphraser`` (GPU) and a spaCy pipeline for the checks.

Outputs
    A list of (twin record, log row) and a stats Counter.

How to run
    Called by ``src/copies/make_twins.py``. Quick look on a GPU:
        from src.copies.paraphrase import Paraphraser
        Paraphraser().generate(["Dubai Metro suspended all lines in 2024"])
"""

from __future__ import annotations

import copy
import re
from collections import Counter

MODEL = "Qwen/Qwen3-8B"
MAX_RETRIES = 2
SYSTEM_PROMPT = (
    "You rewrite fact-check claims. Reword the claim so it has exactly the same meaning. "
    "Keep every name, place, organisation, date and number exactly as written. "
    "Do not add or remove information. Answer with the reworded claim only."
)
ENTITY_LABELS = {"PERSON", "GPE", "LOC", "ORG", "DATE", "EVENT", "NORP",
                 "CARDINAL", "MONEY", "PERCENT", "QUANTITY", "ORDINAL", "TIME"}
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def clean_output(text: str) -> str:
    """Tidy a model answer: first non-empty line, no label prefix, no wrapping quotes."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    line = lines[0] if lines else ""
    line = re.sub(r"^(reworded claim|paraphrase|claim)\s*:\s*", "", line, flags=re.I)
    return line.strip().strip('"“”\'').strip()


def normalise(text: str) -> str:
    """Lower case, letters/digits only: used to tell if the wording really changed."""
    return " ".join(re.findall(r"\w+", text.lower()))


def entity_set(doc) -> set[str]:
    """Lower-cased entity texts of a spaCy Doc (labels in ``ENTITY_LABELS``)."""
    return {ent.text.strip().lower() for ent in doc.ents if ent.label_ in ENTITY_LABELS}


def is_good_paraphrase(claim: str, paraphrase: str, claim_doc, para_doc) -> bool:
    """True if the paraphrase changed the wording but kept entities and numbers."""
    if not paraphrase or normalise(paraphrase) == normalise(claim):
        return False
    if sorted(NUMBER_RE.findall(claim)) != sorted(NUMBER_RE.findall(paraphrase)):
        return False
    return entity_set(claim_doc) == entity_set(para_doc)


class Paraphraser:
    """Qwen3-8B in 4-bit that rewrites claims in batches."""

    def __init__(self, model_name: str = MODEL, batch_size: int = 16):
        """Load the tokenizer and the 4-bit model (needs a GPU and bitsandbytes)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                   bnb_4bit_compute_dtype=torch.float16,
                                                   bnb_4bit_quant_type="nf4"))
        self.model.eval()
        self.batch_size = batch_size

    def prompt(self, claim: str) -> str:
        """Chat prompt for one claim, with Qwen3's thinking mode switched off."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Claim: {claim}"}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True,
                                                  enable_thinking=False)

    def generate(self, claims: list[str], sample: bool = False) -> list[str]:
        """One paraphrase per claim (greedy, or sampled with temperature 0.7)."""
        import torch

        outputs = []
        for start in range(0, len(claims), self.batch_size):
            batch = claims[start:start + self.batch_size]
            enc = self.tokenizer([self.prompt(c) for c in batch], return_tensors="pt",
                                 padding=True).to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **enc, max_new_tokens=96, do_sample=sample,
                    temperature=0.7 if sample else None, top_p=0.9 if sample else None,
                    top_k=None, pad_token_id=self.tokenizer.pad_token_id)
            new_tokens = generated[:, enc["input_ids"].shape[1]:]  # drop the prompt
            outputs += [clean_output(t) for t in
                        self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
            print(f"  paraphrased {min(start + self.batch_size, len(claims))}/{len(claims)}")
        return outputs


def make_paraphrase_twins(records: list[dict], generate, nlp,
                          max_retries: int = MAX_RETRIES) -> tuple[list, Counter]:
    """Paraphrase all records, retrying the ones that fail the checks.

    Args:
        records: source records.
        generate: function (claims, sample: bool) -> paraphrases
            (``Paraphraser.generate`` on Colab; a fake in the tests).
        nlp: spaCy pipeline used for the entity check.
        max_retries: extra sampled attempts for claims that failed.

    Returns:
        ([(twin record, log row), ...] in input order, stats Counter).
    """
    claim_docs = {rec["id"]: doc for rec, doc in
                  zip(records, nlp.pipe([r["claim"] for r in records]))}
    accepted, pending, stats = {}, list(records), Counter()
    for attempt in range(1 + max_retries):
        if not pending:
            break
        texts = generate([r["claim"] for r in pending], attempt > 0)
        still_failing = []
        for rec, text, doc in zip(pending, texts, nlp.pipe(texts)):
            if is_good_paraphrase(rec["claim"], text, claim_docs[rec["id"]], doc):
                twin = copy.deepcopy(rec)
                twin["id"] = f"{rec['id']}_P"
                twin["claim"] = text
                accepted[rec["id"]] = (twin, {"id": twin["id"], "type": "P",
                                              "split": rec["split"], "attempts": attempt + 1})
                stats[f"accepted_attempt_{attempt + 1}"] += 1
            else:
                still_failing.append(rec)
        pending = still_failing
    stats["dropped_failed_checks"] = len(pending)
    return [accepted[r["id"]] for r in records if r["id"] in accepted], stats
