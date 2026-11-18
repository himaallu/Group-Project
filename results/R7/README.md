# R7 · run_all.sh + README (Rathi)

- **What ran:** `bash run_all.sh` (default stages `tests tables`) on a fresh merge of main: 136 tests pass, 8 skipped (GPU/parquet); `src/eval/report.py` rebuilt every table from `results/*/preds_*.jsonl` without errors.
- **Settings:** stages `tests data features twins arabic nli verdict explain demo tables` (`STAGES=all`), each running the exact command the task owner used (R1–R6, Y2–Y3, H3–H5); zero-shot NLI is the default model; seeds 42/43/44; a missing script is skipped with a message. No training, so no W&B run.
- **Tables stage:** gold now also covers AVeriTeC test and X-FACT AR test, so H5's `avt_*` and `xfar_*` predictions are scored (before: 135 "no gold record" warnings per file).
- **README:** pipeline table (zero-shot NLI, trained verdict head), Hub models (reranker, NLI v1/v2, index), reproduce + demo commands, provisional results table from `results/H5/summary.csv` (to re-check at the freeze).
- **Not covered yet:** vision (F1–F5) and baselines (F4) are not in the repo; `src/eval/figures.py` arrives with R8.
