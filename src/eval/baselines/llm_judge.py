"""LLM-judge baseline: Qwen3-8B (4-bit) reads the claim and its snippets and gives a verdict.

What it does
    For each record, Qwen3-8B (thinking off) gets the claim and up to 5 evidence
    snippets [1]-[5] and must answer with one word: SUPPORTED, CONTRADICTED or
    UNVERIFIED. Instead of sampling text, we read the next-token logits of the three
    label words after the prompt and softmax them, which gives probabilities for the
    calibration and AUROC metrics and is deterministic (no seed needed).

Needs a GPU (Kaggle T4 x2 or Colab T4/L4): about 0.3 s per record in batches of 8.

Inputs
    --predict: shared-format records (originals and/or twins; English or Arabic).

Outputs
    --out: ``results/<task>/preds_llm_judge.jsonl`` (CLAUDE.md §5). Checkpointed every
    200 records to ``<out>.partial`` so a disconnect does not lose the run.

How to run (Kaggle / Colab, GPU)
    python -m src.eval.baselines.llm_judge \
        --predict data/processed/en/multifc_test.jsonl data/copies/en/test.jsonl \
        --out results/F4/preds_llm_judge.jsonl
    python -m src.eval.baselines.llm_judge --predict ... --out ... --limit 10   # quick check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.baselines.common import evidence_texts, load_records, to_preds, write_preds
from src.vision.tracking import finish, log_summary, start_run

MODEL = "Qwen/Qwen3-8B"
ANSWERS = ("SUPPORTED", "CONTRADICTED", "UNVERIFIED")  # same order as LABELS
MAX_SNIPPETS = 5
MAX_SNIPPET_CHARS = 400
SYSTEM = ("You are a careful fact-checker. Judge the claim ONLY from the numbered evidence. "
          "Answer with exactly one word: SUPPORTED (the evidence says the claim is true), "
          "CONTRADICTED (the evidence says it is false) or UNVERIFIED (the evidence is "
          "unrelated or not enough).")


def build_prompt(record: dict, tokenizer) -> str:
    """Chat prompt for one record, ending right where the one-word answer starts."""
    snippets = evidence_texts(record)[:MAX_SNIPPETS]
    evidence = "\n".join(f"[{i + 1}] {s[:MAX_SNIPPET_CHARS]}" for i, s in enumerate(snippets)) or "(none)"
    user = f"Claim: {record['claim']}\n\nEvidence:\n{evidence}\n\nVerdict:"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def load_model():
    """Qwen3-8B in 4-bit NF4 on the GPU (same settings as src/explain/generate.py)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    return tok, model.eval()


def answer_token_ids(tokenizer) -> list[int]:
    """First token id of each answer word (the words differ in their first token)."""
    ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in ANSWERS]
    assert len(set(ids)) == len(ids), "answer words must start with different tokens"
    return ids


def judge_batch(records: list[dict], tok, model, answer_ids: list[int]) -> np.ndarray:
    """(n, 3) probabilities from the next-token logits of the three answer words."""
    import torch

    prompts = [build_prompt(r, tok) for r in records]
    inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, :]  # left padding: last position is the next token
    return torch.softmax(logits[:, answer_ids].float(), dim=-1).cpu().numpy()


def main() -> None:
    """Judge every record (resuming from the .partial checkpoint) and write predictions."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predict", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    records = load_records(args.predict)[: args.limit]
    partial = Path(args.out + ".partial")
    done = {}
    if partial.exists():  # resume after a disconnect
        for line in open(partial, encoding="utf-8"):
            row = json.loads(line)
            done[row["id"]] = row["p"]
    todo = [r for r in records if r["id"] not in done]

    tok, model = load_model()
    answer_ids = answer_token_ids(tok)
    partial.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch):
            part = todo[start:start + args.batch]
            for rec, p in zip(part, judge_batch(part, tok, model, answer_ids)):
                done[rec["id"]] = p.tolist()
                f.write(json.dumps({"id": rec["id"], "p": p.tolist()}) + "\n")
            if (start // args.batch) % 25 == 0:
                f.flush()
                print(f"{len(done)}/{len(records)}")

    proba = np.array([done[r["id"]] for r in records])
    write_preds(to_preds(records, proba), args.out)
    run = start_run(f"F4-llm_judge-{Path(args.out).stem}", {"seed": 42, "model": MODEL,
                    "predict": args.predict, "max_snippets": MAX_SNIPPETS}, not args.no_wandb)
    log_summary(run, {"n_predictions": len(records)})
    finish(run)
    print(f"LLM judge: {len(records)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
