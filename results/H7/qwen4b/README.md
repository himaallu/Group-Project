# H7 · Quantized Qwen3-4B vs Qwen3-8B in the team's LLM roles (Himasri)

1. **What ran:** `notebooks/H7_qwen4b_swap_colab.ipynb` on a Colab T4: the team's own scripts, unchanged, with the model swapped to `Qwen/Qwen3-4B` (4-bit NF4) by `src/eval/swap_model.py`: claim extraction (`src.claims.evaluate`, 200 held-out messages), cited explanations (`src.explain.generate`, same 250 claims, seed 42) + faithfulness (`src.explain.faithfulness`, zero-shot NLI), LLM judge (`src.eval.baselines.llm_judge`, MultiFC test + twins and AVeriTeC test). Paired bootstrap 4B vs 8B judge with `src/eval/compare.py`.
2. **Settings:** identical to the 8B runs (Y3, R5, F4) except the model: greedy decoding, thinking off, 5 snippets for the judge, 3 cited snippets for notes. No W&B (team decision).
3. **Claim extraction (F1, 8B → 4B):** all 0.984 → 0.977; EN 0.989 → 0.985, AR 0.966 → 0.970, mixed 1.000 → 0.976 (`claims_f1_qwen4b.json`). Explanations: faithfulness 0.320 → 0.352, citation accuracy 0.306 → 0.363, fully faithful notes 0.208 → 0.180 (`explain/explain_eval.json`; not significance-tested).
4. **LLM judge (8B → 4B):** MultiFC macro-F1 0.233 → 0.391 (+15.8, p < 0.001), ESR 0.100 → 0.580 (+48.0, p < 0.001), unrelated rate 0.950 → 0.136 (−81.4, p < 0.001), paraphrase 0.948 → 0.960 (n.s.), ECE 0.66 → 0.35; AVeriTeC macro-F1 0.494 → 0.613 (+11.9, p < 0.001) — the best AVeriTeC score of any model in the project. Per-class recall on MultiFC: 8B supported 0.73 / contradicted 0.06 / unverified 0.36; 4B 0.58 / 0.61 / 0.01: the 8B judge almost never says contradicted, the 4B judge almost never says unverified (`bootstrap_judge_4b_vs_8b.json`).
5. **Files:** `claims_*`, `notes.jsonl`, `explain/`, `preds_llm_judge4b.jsonl`, `preds_avt_llm_judge4b.jsonl`, `bootstrap_judge_4b_vs_8b.json` in this folder. 8B references: `results/Y3/f1_qwen.json`, `results/R5/explain_eval.json`, `results/F4/preds_*llm_judge.jsonl`.

## Qwen3-4B (4-bit + LoRA) as the verdict model (`src/verdict/llm_verdict.py`, `notebooks/H7_qwen4b_colab.ipynb`)
QLoRA on a Colab T4, seed 42, 1 epoch, LoRA r 16 on all attention + MLP projections, prompt = claim + top-3 reranked snippets, answer = label word (probabilities from the next-token scores of the three labels). `orig` = 4,500 balanced MultiFC train originals (66 min training); `twins` = the same + 1,800 train twins as labelled examples (93 min). Bootstrap: `bootstrap_llm4b.json`.

| Model (MultiFC test + twins) | Macro-F1 | AUROC | ECE | ESR | Unrelated | Paraphrase | AVeriTeC F1 |
|---|---|---|---|---|---|---|---|
| Qwen3-4B fine-tuned, `orig` | **0.500** | 0.790 | 0.098 | 0.306 | 0.148 | 0.787 | 0.574 |
| Qwen3-4B fine-tuned, `twins` | 0.442 | 0.771 | 0.168 | 0.673 | 0.992 | 0.874 | 0.510 |
| MLP + twin loss (H5, seed 42) | 0.344 | | | **0.890** | **1.000** | **0.931** | 0.383 (mean) |
| Claim-only baseline (F4) | 0.481 | | | 0.445 | 0.012 | 0.845 | 0.333 |

- Fine-tuning the 4B model on originals gives the best MultiFC accuracy in the project but fails the evidence tests (ESR 0.31, unrelated 0.15): it learns the claim-only shortcut.
- Adding twins as training data fixes much of it: ESR +36.7, unrelated +84.4, paraphrase +8.6 points (all p < 0.01), for −5.8 macro-F1.
- The small MLP with the twin loss still has the highest ESR (+21.7 over `twins`, p < 0.001) and unrelated rate; `twins` keeps +9.8 macro-F1 over it (p < 0.001). The `twins` LLM is above claim-only on every twin metric (ESR +22.8, unrelated +98.0, p < 0.001) at a similar macro-F1 (−3.9, p = 0.06).
- Seed 42 only (time); adapters in `models/H7/llm4b_{orig,twins}/` (fp16, ~63 MB each).
