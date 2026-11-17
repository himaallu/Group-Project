"""Score (claim, snippet) pairs with the fine-tuned TwinCheck NLI model (R4).

What it does
    ``score_pairs(claims, snippets)`` returns entail / contradict / neutral
    probabilities for each pair. ``src/pipeline.py`` (H4) calls it for its NLI step,
    and R5 uses it for explanation faithfulness. The model is loaded once and kept.
    The CLI scores whole data files and writes the same parquet columns as Himasri's
    ``src/nli/score_zeroshot.py``, so fine-tuned and zero-shot scores can be swapped.
    Input order is the same (premise = snippet, hypothesis = claim), but pairs are cut
    with "longest_first": "only_first" crashes when a claim alone is nearly as long
    as the maximum length (seen on some Arabic pairs in R4).

    Which model: the ``--model`` / ``model`` argument; else the ``TWINCHECK_NLI``
    environment variable; else ``models/nli_v1`` if it holds weights; else the copy on
    Hugging Face Hub (``RathiAbeth/twincheck-nli-v1``; too big for the repo).

    Importing this module needs torch; without it the import fails on purpose (see
    the import below), and the pipeline falls back to its word-overlap stand-in.

Inputs
    score_pairs: claims, snippets (two lists of strings of the same length).
    CLI: JSONL files in the shared record format.

Outputs
    score_pairs -> [{"p_entail", "p_contradict", "p_neutral"}, ...].
    CLI -> <out-dir>/<input stem>.parquet with id, snippet_idx, p_entail, p_contradict, p_neutral.

How to run (repo root)
    python -m src.nli.score data/processed/en/multifc_val.jsonl --limit 10 --out-dir /tmp/nli10
    python -m src.nli.score data/processed/en/multifc_test.jsonl --model models/nli_v1 \
        --out-dir features/nli_v1
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from pathlib import Path

# Without torch this import fails, so src/pipeline.py (which imports this module only
# if it can) keeps its simple stand-in instead of crashing on a laptop with no torch.
import torch

from src.nli.score_zeroshot import OUT_COLUMNS, build_pairs, label_columns

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
def load(name: str):
    """(tokenizer, model, device) for ``name``, loaded once, in eval mode.

    fp32 on purpose: DeBERTa-v3 can overflow in half precision.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
    return tok, model, device


def score_pairs(claims: list[str], snippets: list[str], model: str | None = None,
                batch_size: int = 64, max_len: int = 256) -> list[dict]:
    """NLI probabilities for aligned claims and snippets (premise = snippet).

    Pairs are sorted by length so batches pad little, then put back in order.

    Returns:
        One {"p_entail", "p_contradict", "p_neutral"} dict per pair (4 decimals).
    """
    if len(claims) != len(snippets):
        raise ValueError("claims and snippets must have the same length")
    if not claims:
        return []
    tok, nli, device = load(model or model_name())
    cols = label_columns(nli.config.id2label)
    order = sorted(range(len(claims)), key=lambda i: len(snippets[i]) + len(claims[i]))
    out: list[dict | None] = [None] * len(claims)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            enc = tok([snippets[i] for i in batch], [claims[i] for i in batch],
                      truncation="longest_first", max_length=max_len, padding=True,
                      return_tensors="pt").to(device)
            probs = torch.softmax(nli(**enc).logits.float(), dim=-1).cpu().tolist()
            for i, p in zip(batch, probs):
                out[i] = {col: round(p[j], 4) for col, j in cols.items()}
    return out


def score_file(path: str, out_dir: str, model: str, limit: int | None = None) -> Path:
    """Score every snippet of one data file and write ``<out_dir>/<stem>.parquet``."""
    import pandas as pd

    from src.eval.loaders import load_jsonl

    records = load_jsonl(path)[:limit] if limit else load_jsonl(path)
    pairs = build_pairs(records)
    start = time.time()
    scores = pd.DataFrame(score_pairs(list(pairs["hypothesis"]), list(pairs["premise"]), model),
                          columns=["p_entail", "p_contradict", "p_neutral"])
    table = pd.concat([pairs[["id", "snippet_idx"]].reset_index(drop=True), scores], axis=1)
    out = Path(out_dir) / f"{Path(path).stem}{f'_first{limit}' if limit else ''}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table[list(OUT_COLUMNS)].to_parquet(out, index=False)
    print(f"{path}: {len(records)} records, {len(pairs)} pairs -> {out} "
          f"({time.time() - start:.0f}s)")
    return out


def main() -> None:
    """CLI: score one or more data files."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="JSONL files in the shared record format")
    ap.add_argument("--model", help="default: TWINCHECK_NLI, models/nli_v1 or the Hub copy")
    ap.add_argument("--out-dir", default="features/nli_v1")
    ap.add_argument("--limit", type=int, help="first N records only (10-row check)")
    args = ap.parse_args()
    for path in args.files:
        score_file(path, args.out_dir, args.model or model_name(), args.limit)


if __name__ == "__main__":
    main()
