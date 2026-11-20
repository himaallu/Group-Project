"""Qwen3-4B (4-bit + LoRA) as an end-to-end verdict model, tested with twins (H7).

What it does
    Fine-tunes a quantized LLM to read a claim and its top-3 evidence snippets
    (ranked by Rathi's reranker scores) and answer one word: supported,
    contradicted or unverified. This is the "big model" alternative to our small
    MLP verdict head, asked for by the professor. Two training sets:
        orig    balanced MultiFC train originals only
        twins   the same + train twins as labelled examples (_P keeps the label,
                _B -> contradicted, _C -> unverified)
    Prediction does one forward pass per record and reads the next-token scores of
    the three label words (softmax over those three = the probabilities), so it is
    fast and always gives a valid label. Outputs use the shared prediction format,
    so ``src/eval/metrics.py`` and ``src/eval/compare.py`` score them like any model.

Inputs
    data/processed/en/multifc_{train,test}.jsonl, data/copies/en/{train,test}.jsonl,
    data/processed/en/averitec_test.jsonl,
    features/relevance_en.parquet, features/relevance_copies_en.parquet

Outputs (VARIANT = orig | twins)
    models/H7/llm4b_<VARIANT>/          LoRA adapter (+ tokenizer config)
    results/H7/qwen4b/preds_llm4b_<VARIANT>_s42.jsonl       MultiFC test + twins
    results/H7/qwen4b/preds_avt_llm4b_<VARIANT>_s42.jsonl   AVeriTeC test
    results/H7/qwen4b/metrics_llm4b_<VARIANT>_s42.json      compute_all + settings

How to run
    # 10-row check on CPU with a tiny model, no 4-bit
    python -m src.verdict.llm_verdict --variant orig --model Qwen/Qwen3-0.6B --no-4bit \
        --n-per-class 4 --n-twin-per-type 2 --limit-test 10 --max-steps 3
    # full run on a Colab T4 / L4 (~30-45 min per variant on a T4, ~half on an L4)
    python -m src.verdict.llm_verdict --variant orig
    python -m src.verdict.llm_verdict --variant twins
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.loaders import join_gold, load_jsonl, split_twin_id, twin_type
from src.eval.metrics import LABELS, compute_all

DEFAULT_MODEL = "Qwen/Qwen3-4B"
SYSTEM = ("You are a careful fact-checker. Decide only from the evidence given. "
          "If the evidence says nothing about the claim, the answer is unverified.")
MAX_SNIPPETS, MAX_CLAIM_CHARS, MAX_SNIPPET_CHARS = 3, 300, 350


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

def relevance_lookup(paths: list[str]) -> dict[str, dict[int, float]]:
    """{id: {snippet_idx: relevance}} from R2/R3 relevance files."""
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    out: dict[str, dict[int, float]] = {}
    for rid, idx, rel in zip(df["id"], df["snippet_idx"], df["relevance"]):
        out.setdefault(rid, {})[int(idx)] = float(rel)
    return out


def top_snippets(rec: dict, rel: dict) -> list[str]:
    """The record's MAX_SNIPPETS most relevant snippets (own scores, else the source's
    for _P/_B twins, else file order), each cut to MAX_SNIPPET_CHARS."""
    scores = rel.get(rec["id"])
    if scores is None and twin_type(rec["id"]) in ("P", "B", "A", "M"):
        scores = rel.get(split_twin_id(rec["id"])[0])
    scores = scores or {}
    order = sorted(range(len(rec["evidence"])), key=lambda i: -scores.get(i, 0.0))
    return [rec["evidence"][i]["text"][:MAX_SNIPPET_CHARS] for i in order[:MAX_SNIPPETS]]


def user_message(rec: dict, rel: dict) -> str:
    """Claim + numbered evidence + the answer instruction."""
    snippets = top_snippets(rec, rel)
    evidence = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(snippets)) or "(no evidence)"
    return (f"Claim: {rec['claim'][:MAX_CLAIM_CHARS]}\n\nEvidence:\n{evidence}\n\n"
            "Answer with one word: supported, contradicted or unverified.")


def prompt_text(tok, rec: dict, rel: dict) -> str:
    """Chat-formatted prompt ending where the answer starts (Qwen3 thinking off)."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message(rec, rel)}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def label_token_ids(tok) -> list[int]:
    """First token id of each label word, in LABELS order (must all differ)."""
    ids = [tok(lab, add_special_tokens=False)["input_ids"][0] for lab in LABELS]
    if len(set(ids)) != len(ids):
        raise ValueError(f"label words share a first token: {dict(zip(LABELS, ids))}")
    return ids


# --------------------------------------------------------------------------
# Training data
# --------------------------------------------------------------------------

def sample_training(originals: list[dict], twins: list[dict], variant: str, n_per_class: int,
                    n_twin_per_type: int, seed: int = 42) -> list[dict]:
    """Balanced originals (small classes sampled with replacement) [+ twins], shuffled."""
    rng = random.Random(seed)
    chosen = []
    for lab in LABELS:
        pool = [r for r in originals if r["label"] == lab]
        if not pool:
            continue
        chosen += (rng.sample(pool, n_per_class) if len(pool) >= n_per_class
                   else [rng.choice(pool) for _ in range(n_per_class)])
    if variant == "twins":
        for kind in ("P", "B", "C"):
            pool = [t for t in twins if twin_type(t["id"]) == kind]
            chosen += rng.sample(pool, min(n_twin_per_type, len(pool)))
    rng.shuffle(chosen)
    return chosen


def encode_example(tok, rec: dict, rel: dict, max_len: int) -> dict:
    """input_ids + labels (prompt tokens masked with -100, answer + end token kept)."""
    prompt = tok(prompt_text(tok, rec, rel), add_special_tokens=False)["input_ids"]
    answer = tok(rec["label"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    prompt = prompt[-(max_len - len(answer)):]          # keep the end of a long prompt
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}


def collate(batch: list[dict], pad_id: int):
    """Right-pad a batch for training."""
    import torch
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id)
    labels = torch.full((len(batch), width), -100)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return ids, mask, labels


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def load_base(name: str, four_bit: bool):
    """Tokenizer + causal LM, 4-bit NF4 on GPU (QLoRA) or full precision."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if four_bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            name, device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16,
                                                   bnb_4bit_use_double_quant=True))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    return tok, model


def add_lora(model, r: int = 16):
    """LoRA adapters on the attention and MLP projections."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(model, cfg)


def train(model, tok, examples: list[dict], batch_size: int, grad_accum: int, lr: float,
          max_steps: int | None, seed: int = 42) -> dict:
    """One epoch of LoRA training (fp16 autocast on GPU). Returns loss log."""
    import torch
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
    n_updates = len(batches) // grad_accum
    if max_steps:
        n_updates = min(n_updates, max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(        # 5% warmup, then linear decay
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * n_updates))) * max(0.0, 1 - s / max(1, n_updates)))
    log, step, start = [], 0, time.time()
    for i, batch in enumerate(batches):
        ids, mask, labels = (t.to(device) for t in collate(batch, tok.pad_token_id))
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / grad_accum
        scaler.scale(loss).backward()
        if (i + 1) % grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sched.step()
            step += 1
            log.append(loss.item() * grad_accum)
            if step % 20 == 0 or step == n_updates:
                print(f"step {step}/{n_updates} loss {np.mean(log[-20:]):.4f} "
                      f"({time.time() - start:.0f}s)", flush=True)
            if step >= n_updates:
                break
    return {"steps": step, "final_loss": float(np.mean(log[-20:])) if log else None,
            "train_seconds": round(time.time() - start, 1)}


def predict(model, tok, records: list[dict], rel: dict, batch_size: int = 8) -> list[dict]:
    """Shared-format predictions from next-token scores of the three label words."""
    import torch
    model.eval()
    device = next(model.parameters()).device
    label_ids = label_token_ids(tok)
    tok.padding_side = "left"                       # last position = next token for every row
    out = []
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            enc = tok([prompt_text(tok, r, rel) for r in batch], return_tensors="pt",
                      padding=True, add_special_tokens=False).to(device)
            logits = model(**enc).logits[:, -1, :].float()
            probs = torch.softmax(logits[:, label_ids], dim=-1).cpu().numpy()
            for rec, p in zip(batch, probs):
                p = np.round(p, 6)
                p = p / p.sum()
                out.append({"id": rec["id"], "pred_label": LABELS[int(p.argmax())],
                            "probs": {lab: float(v) for lab, v in zip(LABELS, p)}})
    tok.padding_side = "right"
    return out


def save_adapter(model, folder: Path) -> None:
    """Save only the LoRA weights, cast to fp16 (~66 MB for Qwen3-4B, under GitHub's 100 MB)."""
    import torch
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.to(torch.float16)
    model.save_pretrained(folder)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write dicts as JSON Lines (creates the folder)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> dict:
    """Train one variant, predict MultiFC test + twins and AVeriTeC test, score."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=["orig", "twins"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-4bit", action="store_true", help="full precision (CPU check)")
    ap.add_argument("--n-per-class", type=int, default=1500)
    ap.add_argument("--n-twin-per-type", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--max-steps", type=int, help="stop early (quick check)")
    ap.add_argument("--limit-test", type=int, help="first N test records (quick check)")
    ap.add_argument("--en-dir", default="data/processed/en")
    ap.add_argument("--copies-dir", default="data/copies/en")
    ap.add_argument("--relevance", nargs="+",
                    default=["features/relevance_en.parquet", "features/relevance_copies_en.parquet"])
    ap.add_argument("--out-dir", default="results/H7/qwen4b")
    ap.add_argument("--models-dir", default="models/H7")
    args = ap.parse_args(argv)

    random.seed(42)
    rel = relevance_lookup(args.relevance)
    originals = load_jsonl(f"{args.en_dir}/multifc_train.jsonl")
    twins = load_jsonl(f"{args.copies_dir}/train.jsonl")
    train_set = sample_training(originals, twins, args.variant, args.n_per_class,
                                args.n_twin_per_type)
    test = load_jsonl(f"{args.en_dir}/multifc_test.jsonl") + load_jsonl(f"{args.copies_dir}/test.jsonl")
    avt = load_jsonl(f"{args.en_dir}/averitec_test.jsonl")
    if args.limit_test:
        test, avt = test[:args.limit_test], avt[:args.limit_test]

    tok, model = load_base(args.model, four_bit=not args.no_4bit)
    model = add_lora(model)
    examples = [encode_example(tok, r, rel, args.max_len) for r in train_set]
    print(f"{args.variant}: {len(examples)} training examples "
          f"({sum(1 for r in train_set if twin_type(r['id']))} twins)", flush=True)
    log = train(model, tok, examples, args.batch_size, args.grad_accum, args.lr, args.max_steps)

    name = f"llm4b_{args.variant}"
    out = Path(args.out_dir)
    preds = predict(model, tok, test, rel)
    avt_preds = predict(model, tok, avt, rel)
    write_jsonl(preds, out / f"preds_{name}_s42.jsonl")
    write_jsonl(avt_preds, out / f"preds_avt_{name}_s42.jsonl")
    df, _ = join_gold(preds, test)
    avt_df, _ = join_gold(avt_preds, avt)
    metrics = {"multifc": compute_all(df), "averitec": compute_all(avt_df),
               "settings": {k: v for k, v in vars(args).items()}, "train": log,
               "n_train": len(examples)}
    (out / f"metrics_{name}_s42.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_adapter(model, Path(args.models_dir) / name)
    m = metrics["multifc"]
    print(f"{name}: macro-F1 {m['macro_f1']:.3f} ESR {m['esr']} unrelated {m['unrelated_rate']} "
          f"paraphrase {m['paraphrase_consistency']} | AVeriTeC macro-F1 "
          f"{metrics['averitec']['macro_f1']:.3f}", flush=True)
    return metrics


if __name__ == "__main__":
    main()
