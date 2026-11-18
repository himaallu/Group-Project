# F4: Baselines audit E0 (Feba)

1. **What ran:**
   - Claim-only, similarity + LightGBM and zero-shot NLI baselines (`src/eval/baselines/`) on the MultiFC test originals (2,106) and all English twins (174 P, 281 B, 500 C), and separately on the AVeriTeC test set (429, `preds_avt_*`).
   - `src/eval/baselines/audit.py` → `audit_E0.csv`.
   - The LLM judge (Qwen3-8B, 4-bit) needs a GPU. Run it with `notebooks/F4_llm_judge_colab.ipynb`. **Not run yet.**
2. **Settings:**
   - All models are trained on MultiFC train, and their settings are chosen on MultiFC val.
   - Claim-only: TF-IDF 1–2-grams + logistic regression, C = 10.
   - Similarity: 12 lexical features + LightGBM, seeds 42/43/44.
   - Zero-shot NLI: rule on max entail / contradict over snippets with an abstain threshold, t = 0.05 chosen on val. It uses the scores in `features/nli_zeroshot/`.
3. **W&B:** runs `F4-claim_only-s42` and `F4-similarity-s4*`, logged offline on Feba's laptop; they need `wandb sync`. The LLM judge logs from Colab.
4. **Key numbers** (`audit_E0.csv`, MultiFC; sd over 3 seeds for similarity):

   | Baseline | Macro-F1 | ESR | Unrelated rate | Paraphrase consistency |
   |---|---|---|---|---|
   | Claim-only | 0.481 | 0.445 | 0.012 | 0.845 |
   | Similarity | 0.385 ± 0.005 | 0.592 ± 0.011 | 0.014 | 0.730 |
   | Zero-shot NLI | 0.346 | 0.623 | 0.536 | 0.874 |

   AVeriTeC macro-F1: claim-only 0.333, similarity 0.281, zero-shot NLI 0.440.
5. **Where files are:**
   - Predictions are in `results/F4/preds_*.jsonl`, and the table is `results/F4/audit_E0.csv`.
   - To run the scripts on Arabic twins (Y5), pass the X-FACT AR train/val files with `--train/--val` and the Arabic twin file with `--predict`. For zero-shot NLI, also pass the matching `features/nli_zeroshot/*.parquet`.
