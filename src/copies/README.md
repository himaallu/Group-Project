# src/copies: twins

| File | Owner | What it does |
|---|---|---|
| `sources.py` | Rathi (R3) | picks twin sources (supported, ≥ 2 snippets with relevance > 0.3, seed 42) |
| `swap.py` | Rathi (R3) | `_B`: swaps one place / person / year in every snippet → contradicted |
| `unrelated.py` | Rathi (R3) | `_C`: evidence of an unrelated claim → unverified |
| `paraphrase.py` | Rathi (R3) | `_P`: Qwen3-8B paraphrase, kept only if entities and numbers match |
| `make_twins.py` | Rathi (R3) | runs all of the above → `data/copies/en/` |
| `checks.py` | Rathi (R3) | automatic checks on every twin + 10 samples per type |
| `handcheck_sheet.py`, `handcheck_kappa.py` | Himasri (H4) | hand-check sheets and Cohen's kappa |

Run: `python -m src.copies.make_twins && python -m src.copies.checks` (GPU for `_P`). Tests: `tests/copies/`.
