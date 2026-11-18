# src/nli: natural language inference (entail / contradict / neutral)

| File | Owner | What it does |
|---|---|---|
| `sample_gold.py`, `sheet_to_jsonl.py` | Himasri (H2) | gold-pair sheet → `data/nli/gold_val.jsonl` |
| `score_zeroshot.py` | Himasri (H3) | scores every (claim, snippet) pair with any NLI model → `features/nli_*/` |
| `build_ar_pairs.py` | Youssef (Y2) | Arabic weak pairs from X-FACT |
| `build_pairs.py` | Rathi (R4) | balanced training pairs (train splits only, `_B` swaps as contradictions); v2 keeps a weak pair only if the zero-shot model agrees (`--filter-model`) and adds own-snippet hard neutrals |
| `train.py` | Rathi (R4) | fine-tunes mDeBERTa-v3 (or XLM-R-large), `--version v1/v2`; gold macro-F1 before / after |
| `score.py` | Rathi (R4) | `score_pairs(claims, snippets)` used by the pipeline; default model = zero-shot mDeBERTa (v2 tied with it), set `TWINCHECK_NLI` to change |
| `gold_check.py` | Himasri (H5) | scores any NLI feature files on the gold pairs |

Tests: `tests/nli/`.
