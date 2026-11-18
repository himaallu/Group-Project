# H3 · Verdict head v0 (Himasri)

1. **What ran:** `notebooks/H3_verdict_head_colab.ipynb` on Colab (T4): `src/nli/score_zeroshot.py` (base mDeBERTa-v3 XNLI, no fine-tuning) on all MultiFC snippets → `features/nli_zeroshot/`; then `src/verdict/train.py --name v0` (features from `src/verdict/features.py` + R2 relevance/entities, MLP 10→64→3) for seeds 42/43/44.
2. **Settings:** v0 = no twin terms (λ_inv = λ_sens = 0); class-weighted CE on originals; Adam lr 1e-3, batch 64 groups, ≤ 30 epochs, early stopping on val macro-F1 (patience 5); NLI max len 256 (longest-first truncation).
3. **W&B:** not used (team decision); settings and metrics per run in `metrics_v0_s<seed>.json`, seed table in `summary_v0.csv`.
4. **Key numbers (mean ± sd, 3 seeds):** test macro-F1 **0.363 ± 0.005**, AUROC 0.611 ± 0.003, ECE 0.054 ± 0.001 (val macro-F1 0.373 ± 0.005). Baselines on test: always "contradicted" 0.257, random by class prior 0.328. Seed 42 predicts "unverified" 386× vs 76 gold (precision 0.05): `v0_s42_per_class.json`.
5. **Files:** `preds_v0_s*.jsonl` (test, §5 format), `val_logits_v0_s*.jsonl` (for `calibrate.py`), models in `models/verdict_v0_s*/`, zero-shot NLI features in `features/nli_zeroshot/multifc_{train,val,test}.parquet`.
