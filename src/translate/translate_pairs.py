"""Top up the Arabic NLI training data with translated English entail pairs (task Y2, part 2).

What it does
    X-FACT Arabic has very few true claims, so it gives only ~46 entail pairs.
    This script:
      1. picks English entail pairs from MultiFC train: a supported claim + its snippet
         with the most word overlap (overlap >= 0.5), seed 42;
      2. translates claim and snippet to Modern Standard Arabic with Qwen3-8B
         (4-bit, greedy, thinking off). The prompt keeps names, numbers and dates
         unchanged in meaning; an optional glossary (CSV en,ar) fixes how names are
         written, so the same person is spelled the same way everywhere;
      3. --merge: drops translations that drifted into another script (Cyrillic,
         Chinese) or stayed mostly English, then joins the X-FACT weak pairs and the
         good translated pairs into data/nli/ar_train.jsonl, keeping classes balanced.
    Translation writes one line per finished pair, so a re-run after a Colab
    disconnect continues where it stopped.

Inputs
    data/processed/en/multifc_train.jsonl, data/nli/ar_train_xfact.jsonl,
    optional data/translate/glossary_en_ar.csv

Outputs
    data/nli/ar_train_translated.jsonl   translated pairs (+ "claim_en", "snippet_en")
    data/nli/ar_train.jsonl              final Arabic training pairs for Rathi's R4,
                                         same fields as data/nli/gold_val.jsonl

How to run (repo root, GPU runtime)
    python -m src.translate.translate_pairs --n 10            # test on 10 pairs
    python -m src.translate.translate_pairs --n 954           # full (~30-60 min on a T4)
    python -m src.translate.translate_pairs --merge
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter

from src.data.common import SEED, read_jsonl, write_jsonl

MODEL = "Qwen/Qwen3-8B"
WORD = re.compile(r"\w+", re.UNICODE)
MIN_OVERLAP = 0.5
MAX_SNIPPET_WORDS = 120  # long snippets are cut so the translation stays fast

PROMPT = (
    "Translate the following English text into Modern Standard Arabic.\n"
    "Rules: keep the exact meaning; keep every name, number and date (write numbers "
    "as digits); do not add, explain or drop anything; output only the translation.\n"
    "{glossary}\nText:\n{text}"
)


def overlap(claim: str, snippet: str) -> float:
    """Share of the claim's distinct words that also appear in the snippet."""
    cw = set(WORD.findall(claim.lower()))
    return len(cw & set(WORD.findall(snippet.lower()))) / len(cw) if cw else 0.0


def select_entail_pairs(records: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """Pick n English entail pairs (supported claim + best-overlap snippet).

    Args:
        records: MultiFC train records in the shared format.
        n: number of pairs wanted.
    Returns:
        list of pair dicts in the gold_val.jsonl format (English).
    """
    pairs = []
    for rec in records:
        if rec["label"] != "supported" or not rec.get("evidence"):
            continue
        scores = [(overlap(rec["claim"], s["text"]), i) for i, s in enumerate(rec["evidence"])]
        best, idx = max(scores)
        if best >= MIN_OVERLAP:
            snippet = " ".join(rec["evidence"][idx]["text"].split()[:MAX_SNIPPET_WORDS])
            pairs.append({"id": f"{rec['id']}_s{idx}_tr", "claim_id": rec["id"], "snippet_idx": idx,
                          "language": "en", "claim": rec["claim"], "snippet": snippet,
                          "label": "entail"})
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


def load_glossary(path: str | None) -> str:
    """Return glossary lines for the prompt ('English -> Arabic'), or '' if no file."""
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    lines = [f"{en} -> {ar}" for en, ar in rows[1:] if en and ar]
    return "Use these spellings:\n" + "\n".join(lines) + "\n" if lines else ""


def load_model():
    """Load Qwen3-8B in 4-bit (fits a free T4) with left padding for batched generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16))
    return tok, model


def translate_batch(texts: list[str], tok, model, glossary: str, max_new_tokens: int = 320) -> list[str]:
    """Translate a batch of English texts to Arabic (greedy decoding, thinking disabled)."""
    import torch

    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(glossary=glossary, text=t)}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[:, enc["input_ids"].shape[1]:]
    return [re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
            for t in tok.batch_decode(new, skip_special_tokens=True)]


def run_translation(args) -> None:
    """Translate the selected pairs, appending to the output file (resumable)."""
    pairs = select_entail_pairs(read_jsonl(args.multifc), args.n)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8") if l.strip()}
    todo = [p for p in pairs if p["id"] not in done]
    print(f"selected {len(pairs)} pairs, {len(done)} already done, {len(todo)} to translate")
    if not todo:
        return
    tok, model = load_model()
    glossary = load_glossary(args.glossary)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start + args.batch]
            texts = [p["claim"] for p in batch] + [p["snippet"] for p in batch]
            ar = translate_batch(texts, tok, model, glossary)
            for p, claim_ar, snip_ar in zip(batch, ar[:len(batch)], ar[len(batch):]):
                f.write(json.dumps({**p, "language": "ar", "claim": claim_ar, "snippet": snip_ar,
                                    "claim_en": p["claim"], "snippet_en": p["snippet"],
                                    "origin": "multifc_translated"}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{min(start + args.batch, len(todo))}/{len(todo)} translated")


ARABIC = re.compile(r"[\u0600-\u06FF]")
OTHER_SCRIPT = re.compile(r"[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF]")  # Cyrillic, Chinese, Japanese


def good_translation(p: dict) -> bool:
    """False when the model drifted into another script or left the text mostly untranslated."""
    for text in (p["claim"], p["snippet"]):
        if OTHER_SCRIPT.search(text) or len(ARABIC.findall(text)) < 0.3 * len(text):
            return False
    return True


def merge(args) -> None:
    """Join X-FACT weak pairs and good translated pairs into data/nli/ar_train.jsonl (balanced)."""
    rng = random.Random(SEED)
    translated = read_jsonl(args.out)
    kept = [p for p in translated if good_translation(p)]
    print(f"translated pairs: {len(translated)}, kept {len(kept)} "
          f"(dropped {len(translated) - len(kept)} with another script or little Arabic)")
    pairs = read_jsonl(args.xfact_pairs) + kept
    by_label = {lab: [p for p in pairs if p["label"] == lab] for lab in ("entail", "contradict", "neutral")}
    cap = max(len(by_label["entail"]), 1)  # entail is the smallest class after the top-up
    final = []
    for lab, items in by_label.items():
        rng.shuffle(items)
        final += items[:cap]
    rng.shuffle(final)
    write_jsonl(final, args.final)
    print(f"{args.final}: {len(final)} pairs {dict(Counter(p['label'] for p in final))} "
          f"origins {dict(Counter(p.get('origin') for p in final))}")


def main() -> None:
    """Command-line entry point: translate (default) or --merge."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--multifc", default="data/processed/en/multifc_train.jsonl")
    ap.add_argument("--xfact_pairs", default="data/nli/ar_train_xfact.jsonl")
    ap.add_argument("--out", default="data/nli/ar_train_translated.jsonl")
    ap.add_argument("--final", default="data/nli/ar_train.jsonl")
    ap.add_argument("--glossary", default="data/translate/glossary_en_ar.csv")
    ap.add_argument("--n", type=int, default=954)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    merge(args) if args.merge else run_translation(args)


if __name__ == "__main__":
    main()
