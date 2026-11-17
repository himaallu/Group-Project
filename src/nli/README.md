# src/nli: natural language inference (entail / contradict / neutral)

| File | Owner | What it does |
|---|---|---|
| `sample_gold.py`, `sheet_to_jsonl.py` | Himasri (H2) | gold-pair sheet → `data/nli/gold_val.jsonl` |
| `score_zeroshot.py` | Himasri (H3) | scores every (claim, snippet) pair with any NLI model → `features/nli_*/` |
| `build_ar_pairs.py` | Youssef (Y2) | Arabic weak pairs from X-FACT |
| `build_pairs.py` | Rathi (R4) | balanced weak training pairs (train splits only, `_B` swaps as contradictions) |
| `train.py` | Rathi (R4) | fine-tunes mDeBERTa-v3 (or XLM-R-large); gold macro-F1 before / after |
| `score.py` | Rathi (R4) | `score_pairs(claims, snippets)` with the fine-tuned model, used by the pipeline |

Tests: `tests/nli/`.
