# src/explain: cited notes

| File | Owner | What it does |
|---|---|---|
| `generate.py` | Rathi (R5) | `generate(claim, label, snippets, lang)`: 1–2 sentences citing [1]–[3] (Qwen3-8B; template without a GPU) |
| `faithfulness.py` | Rathi (R5) | NLI check of every sentence against the snippet it cites → faithfulness, citation accuracy |

Run: `python -m src.explain.generate && python -m src.explain.faithfulness` (GPU). Tests: `tests/explain/`.
