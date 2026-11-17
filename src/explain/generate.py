"""Cited explanations: a 1-2 sentence note that cites the evidence as [1]-[3] (R5).

What it does
    ``generate(claim, label, snippets, lang)`` writes a short note for a verdict,
    using only the given snippets (at most 3) and citing them as [1], [2], [3].
    The note is in the claim's language (Arabic for "ar" and "mixed" claims).
    Qwen3-8B (4-bit, thinking off, greedy) writes the note; ``clean_note`` then
    removes any citation outside [1]..[n] and keeps at most 2 sentences.
    The prompt asks the model to state what the evidence says rather than repeat the
    verdict word, so each sentence can be checked against the snippet it cites
    (``src/explain/faithfulness.py``).
    Without a GPU (or with TWINCHECK_EXPLAINER=template) a simple template note that
    quotes the first snippet is returned instead, so the pipeline runs on a laptop.
    ``src/pipeline.py`` (H4) calls ``generate``; the CLI makes the notes for R5's
    evaluation.

Inputs
    claim (str), label (supported / contradicted / unverified), snippets (list of
    {"text", ...} or str), lang ("en" / "ar" / "mixed").
    CLI: --en-files data/processed/en/multifc_test.jsonl  --ar-files data/processed/ar/xfact_ar_test.jsonl

Outputs
    generate -> note (str).
    CLI -> results/R5/notes.jsonl, one row per claim:
    {"id", "language", "claim", "label", "snippets": [3 x {"text","url","domain"}], "note"}
    The verdict explained is the gold label (the verdict head is not ready yet).

How to run (Colab GPU, repo root)
    python -m src.explain.generate --n-en 5 --n-ar 5 --out results/R5/test10/notes.jsonl   # 10-row check
    python -m src.explain.generate --n-en 200 --n-ar 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

MODEL = "Qwen/Qwen3-8B"
MAX_CITED = 3
MAX_SENTENCES = 2
LABEL_WORDS = {
    "en": {"supported": "supported (true)", "contradicted": "contradicted (false)",
           "unverified": "unverified (the evidence does not settle it)"},
    "ar": {"supported": "صحيح", "contradicted": "خاطئ", "unverified": "غير مؤكد"},
}
SYSTEM_PROMPT = (
    "You write short fact-check notes. You get a claim, its verdict and numbered evidence "
    "snippets. Write {n_sent} in {language} that explain the verdict using ONLY facts stated "
    "in the evidence. End every sentence with the number of the snippet it uses, like [1]. "
    "Cite only the numbers given. Do not add facts, do not guess, and do not just repeat the "
    "verdict: say what the evidence shows. If the verdict is unverified, say what the "
    "evidence is about and that it does not confirm or deny the claim. "
    "Output only the note."
)
CITATION_RE = re.compile(r"\[(\d+)\]")
SENTENCE_END_RE = re.compile(r"(?<=[.!?؟])\s+")
LEADING_CITATIONS_RE = re.compile(r"^((?:\[\d+\]\s*)+)")
_model = None


def backend() -> str:
    """'qwen' on a GPU (or if TWINCHECK_EXPLAINER=qwen), otherwise 'template'."""
    forced = os.environ.get("TWINCHECK_EXPLAINER")
    if forced in ("qwen", "template"):
        return forced
    try:
        import torch
        return "qwen" if torch.cuda.is_available() else "template"
    except ImportError:
        return "template"


def snippet_text(snippet) -> str:
    """Text of a snippet given as a dict or a string."""
    return snippet if isinstance(snippet, str) else snippet["text"]


def note_language(lang: str) -> str:
    """'ar' for Arabic and mixed claims, else 'en'."""
    return "ar" if lang in ("ar", "mixed") else "en"


def build_messages(claim: str, label: str, snippets: list, lang: str) -> list[dict]:
    """Chat messages for the model: instructions, then claim, verdict and evidence."""
    out_lang = note_language(lang)
    system = SYSTEM_PROMPT.format(n_sent="1 or 2 short sentences",
                                  language="Modern Standard Arabic" if out_lang == "ar" else "English")
    evidence = "\n".join(f"[{i + 1}] {snippet_text(s)}" for i, s in enumerate(snippets))
    user = (f"Claim: {claim}\nVerdict: {LABEL_WORDS[out_lang][label]}\n"
            f"Evidence:\n{evidence}\n\nNote:")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def split_sentences(note: str) -> list[str]:
    """Split a note into sentences; citations after a full stop join the sentence before.

    "Trains ran normally. [1] Rain fell [2]." -> ["Trains ran normally. [1]", "Rain fell [2]."]
    """
    pieces = [p.strip() for p in SENTENCE_END_RE.split(note.strip()) if p.strip()]
    sentences = []
    for piece in pieces:
        lead = LEADING_CITATIONS_RE.match(piece)
        if sentences and lead:
            sentences[-1] += " " + lead.group(1).strip()
            rest = piece[lead.end():].strip()
            piece = rest if rest.strip(" .") else ""   # only a citation (and a dot) was left
        if piece:
            sentences.append(piece)
    return sentences


def clean_note(text: str, n_snippets: int) -> str:
    """Remove thinking tags, citations outside [1]..[n], extra sentences and spaces."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^(note|الملاحظة)\s*:\s*", "", text, flags=re.I)
    text = CITATION_RE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_snippets else "",
                           text)
    text = " ".join(split_sentences(text)[:MAX_SENTENCES])
    return re.sub(r"\s+", " ", text).replace(" .", ".").strip()


def template_note(label: str, snippets: list, lang: str) -> str:
    """Note without an LLM: the verdict and a short quote of snippet [1]."""
    if not snippets:
        return ("لم يُعثر على أدلة ذات صلة." if note_language(lang) == "ar"
                else "No relevant evidence was found for this claim.")
    quote = snippet_text(snippets[0])[:200].strip()
    if note_language(lang) == "ar":
        return f"الحكم: {LABEL_WORDS['ar'][label]}. الدليل: \"{quote}\" [1]"
    return f"Verdict: {label}. Evidence: \"{quote}\" [1]"


def _load():
    """Load Qwen3-8B once (4-bit on GPU)."""
    global _model
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16))
        _model = (tok, model)
    return _model


def generate_batch(items: list[dict], max_new_tokens: int = 160) -> list[str]:
    """Notes for several {"claim", "label", "snippets", "lang"} items at once."""
    items = [dict(it, snippets=list(it["snippets"])[:MAX_CITED]) for it in items]
    if backend() == "template":
        return [template_note(it["label"], it["snippets"], it["lang"]) for it in items]
    import torch

    tok, model = _load()
    notes = [None] * len(items)
    todo = [i for i, it in enumerate(items) if it["snippets"]]
    for i, it in enumerate(items):
        if not it["snippets"]:
            notes[i] = template_note(it["label"], [], it["lang"])
    if todo:
        prompts = [tok.apply_chat_template(build_messages(items[i]["claim"], items[i]["label"],
                                                          items[i]["snippets"], items[i]["lang"]),
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for i in todo]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, text in zip(todo, texts):
            notes[i] = clean_note(text, len(items[i]["snippets"])) or \
                template_note(items[i]["label"], items[i]["snippets"], items[i]["lang"])
    return notes


def generate(claim: str, label: str, snippets: list, lang: str = "en") -> str:
    """Short cited note for one claim (interface used by src/pipeline.py)."""
    return generate_batch([{"claim": claim, "label": label, "snippets": snippets,
                            "lang": lang}])[0]


def sample_claims(paths: list[str], n: int, seed: int = 42) -> list[dict]:
    """``n`` records with at least one snippet, picked with ``seed`` (sorted by id first)."""
    from src.eval.loaders import load_jsonl

    records = sorted((r for p in paths for r in load_jsonl(p)
                      if any(ev["text"].strip() for ev in r["evidence"])), key=lambda r: r["id"])
    random.Random(seed).shuffle(records)
    return records[:n]


def top_snippets(records: list[dict], reranker, k: int = MAX_CITED) -> list[list[dict]]:
    """The ``k`` snippets of each record that the reranker finds most relevant."""
    out = []
    for rec in records:
        evidence = [ev for ev in rec["evidence"] if ev["text"].strip()]
        scores = reranker.score(rec["claim"], [ev["text"] for ev in evidence])
        order = sorted(range(len(evidence)), key=lambda i: -scores[i])[:k]
        out.append([{"text": evidence[i]["text"], "url": evidence[i].get("url"),
                     "domain": evidence[i].get("domain")} for i in order])
    return out


def main() -> None:
    """CLI: notes for a seed-42 sample of English and Arabic test claims."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--en-files", nargs="+", default=["data/processed/en/multifc_test.jsonl"])
    ap.add_argument("--ar-files", nargs="+", default=["data/processed/ar/xfact_ar_test.jsonl"])
    ap.add_argument("--n-en", type=int, default=200)
    ap.add_argument("--n-ar", type=int, default=50)
    ap.add_argument("--reranker", default="RathiAbeth/twincheck-reranker-v1")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/R5/notes.jsonl")
    args = ap.parse_args()

    from src.retrieval.rerank import Reranker

    records = (sample_claims(args.en_files, args.n_en, args.seed)
               + sample_claims(args.ar_files, args.n_ar, args.seed))
    snippets = top_snippets(records, Reranker(args.reranker))
    rows = [{"id": r["id"], "language": r["language"], "claim": r["claim"], "label": r["label"],
             "snippets": s} for r, s in zip(records, snippets)]
    print(f"{len(rows)} claims, backend {backend()}")
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        notes = generate_batch([{"claim": r["claim"], "label": r["label"],
                                 "snippets": r["snippets"], "lang": r["language"]} for r in batch])
        for row, note in zip(batch, notes):
            row["note"] = note
        print(f"  {start + len(batch)}/{len(rows)}: {batch[0]['note'][:100]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
