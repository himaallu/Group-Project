# R4 · NLI fine-tuning (Rathi)

- **What ran:** `src/nli/build_pairs.py` (v1: 32,658 weak pairs = 30k EN + 2,658 AR, labels copied from the claim label) → `src/nli/train.py` on mDeBERTa-v3-base XNLI (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`), plus XLM-R-large XNLI as comparison; evaluated on the 300 gold pairs (150 EN, 150 AR).
- **Settings:** lr 2e-5 (XLM-R 1e-5), batch 16, max len 256, 2 epochs, warmup 10%, fp32/TF32, seed 42, NVIDIA L4 (19 min mDeBERTa, 33 min XLM-R).
- **W&B:** mDeBERTa https://forge.coreweave.com/wandb/rathiabeth-uowd/twincheck/runs/gepd51e6 · XLM-R https://forge.coreweave.com/wandb/rathiabeth-uowd/twincheck/runs/e50h261l
- **Key numbers (v1, gold macro-F1 zero-shot → fine-tuned):** mDeBERTa EN 0.371 → 0.124, AR 0.443 → 0.162; XLM-R EN 0.382 → 0.124, AR 0.502 → 0.174; weak val 0.40 → 0.76. v1 copies the claim label onto on-topic snippets (gold pairs are 75% neutral), so **zero-shot is the default NLI model** (`src/nli/score.py`). v2 (agreement filter + hard neutrals, `--filter-model`) is coded and tested; its run is pending.
- **Files:** `results/R4/v1_{gold_eval,gold_eval_xlmr,pairs_report}.json`, `data/nli/v1/{train,val_weak}.jsonl`, `features/nli_v1/*.parquet` (v1 scores); weights on the HF Hub `RathiAbeth/twincheck-nli-v1` (not for use).
