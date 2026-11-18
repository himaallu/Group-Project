"""Zero-shot NLI scores for every (claim, snippet) pair with the base mDeBERTa model (H3 step 1).

What it does
    Runs an off-the-shelf multilingual NLI model (no fine-tuning) on every snippet of
    every record: premise = snippet text, hypothesis = claim. Gives the verdict head
    its zero-shot NLI features (H3 v0) and the "zero-shot NLI" ablation (H5).
    Rathi's fine-tuned scores (R4) use the same columns, so the two are swappable.

Inputs
    One or more JSONL files in the shared record format, e.g.
    data/processed/en/multifc_{train,val,test}.jsonl or data/copies/en/test.jsonl.

Outputs
    One parquet file per input under ``--out-dir`` (default
    ``features/nli_zeroshot/``), named after the input, e.g.
    ``features/nli_zeroshot/multifc_val.parquet``, with columns
        id, snippet_idx, p_entail, p_contradict, p_neutral
    (one row per snippet; records with no snippets have no rows).

How to run
    # 10-row check first (CLAUDE.md rule 3)
    python -m src.nli.score_zeroshot data/processed/en/multifc_val.jsonl --limit 10
    # full run on a Colab GPU (about 15-25 min for all MultiFC splits on a T4)
    python -m src.nli.score_zeroshot data/processed/en/multifc_{train,val,test}.jsonl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.eval.loaders import load_jsonl

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
OUT_COLUMNS = ("id", "snippet_idx", "p_entail", "p_contradict", "p_neutral")


def build_pairs(records: list[dict]) -> pd.DataFrame:
    """One row per (record, snippet): id, snippet_idx, premise (snippet), hypothesis (claim).

    Empty snippets are skipped.
    """
    rows = []
    for rec in records:
        for idx, ev in enumerate(rec.get("evidence") or []):
            text = (ev.get("text") or "").strip()
            if text:
                rows.append({"id": rec["id"], "snippet_idx": idx,
                             "premise": text, "hypothesis": rec["claim"]})
    return pd.DataFrame(rows, columns=["id", "snippet_idx", "premise", "hypothesis"])


def label_columns(id2label: dict) -> dict[str, int]:
    """Map the model's output positions to our three NLI columns.

    Args:
        id2label: ``model.config.id2label``, e.g. {0: "entailment", 1: "neutral",
            2: "contradiction"}. Order differs between checkpoints, so never assume it.

    Returns:
        {"p_entail": i, "p_contradict": j, "p_neutral": k}

    Raises:
        ValueError: if the label names do not show all three classes.
    """
    found = {}
    for idx, name in id2label.items():
        name = str(name).lower()
        if name.startswith("entail"):
            found["p_entail"] = int(idx)
        elif name.startswith("contra"):
            found["p_contradict"] = int(idx)
        elif name.startswith("neutral"):
            found["p_neutral"] = int(idx)
    if len(found) != 3:
        raise ValueError(f"cannot read NLI labels from id2label={id2label}")
    return found


def load_model(name: str = DEFAULT_MODEL, device: str | None = None):
    """Load tokenizer + model in eval mode on GPU if available.

    Returns:
        (tokenizer, model, device string).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
    if device == "cuda":
        model = model.half()  # fp16: about 2x faster on a T4, same scores to ~3 decimals
    return tok, model, device


def score_pairs(premises: list[str], hypotheses: list[str], tok, model, device: str,
                batch_size: int = 32, max_len: int = 256) -> pd.DataFrame:
    """NLI probabilities for aligned lists of premises and hypotheses.

    Pairs are sorted by length so each batch pads as little as possible, then put
    back in the original order.

    Returns:
        DataFrame with p_entail, p_contradict, p_neutral (same order as the input).
    """
    import torch

    cols = label_columns(model.config.id2label)
    order = sorted(range(len(premises)), key=lambda i: len(premises[i]) + len(hypotheses[i]))
    probs = [None] * len(premises)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            enc = tok([premises[i] for i in batch], [hypotheses[i] for i in batch],
                      # cut the longer side token by token: normally the snippet; a few
                      # claims are longer than max_len on their own, then the claim is cut too
                      truncation="longest_first", max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            p = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
            for i, row in zip(batch, p):
                probs[i] = row
    out = pd.DataFrame({col: [float(row[j]) for row in probs] for col, j in cols.items()})
    return out[["p_entail", "p_contradict", "p_neutral"]]


def score_file(path: str | Path, tok, model, device: str, out_dir: str | Path,
               limit: int | None = None, batch_size: int = 32, max_len: int = 256) -> Path:
    """Score every snippet of one data file and write the parquet output.

    Args:
        path: input JSONL (shared record format).
        out_dir: output folder; the file is named ``<input stem>.parquet``.
        limit: only the first ``limit`` records (for the 10-row check).

    Returns:
        Path of the written parquet file.
    """
    records = load_jsonl(path)[:limit] if limit else load_jsonl(path)
    pairs = build_pairs(records)
    start = time.time()
    scores = score_pairs(list(pairs["premise"]), list(pairs["hypothesis"]), tok, model,
                         device, batch_size, max_len) if len(pairs) else \
        pd.DataFrame(columns=["p_entail", "p_contradict", "p_neutral"])
    result = pd.concat([pairs[["id", "snippet_idx"]].reset_index(drop=True), scores], axis=1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem + (f"_first{limit}" if limit else "")
    out = out_dir / f"{stem}.parquet"
    result[list(OUT_COLUMNS)].to_parquet(out, index=False)
    print(f"{path}: {len(records)} records, {len(pairs)} pairs -> {out} "
          f"({time.time() - start:.1f}s on {device})")
    return out


def main() -> None:
    """CLI: score one or more data files."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="JSONL files in the shared format")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", default="features/nli_zeroshot")
    ap.add_argument("--limit", type=int, help="first N records only (10-row check)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args()
    tok, model, device = load_model(args.model, args.device)
    for path in args.files:
        score_file(path, tok, model, device, args.out_dir, args.limit,
                   args.batch_size, args.max_len)


if __name__ == "__main__":
    main()
