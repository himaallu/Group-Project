# R9 · Technical appendix + code hygiene (Rathi)

- **What ran:** filled `docs/implementation.md` (environment, data, reranker, twins, NLI v1/v2 with the RCA, explanations + index, Hub files, demo, teammates' components) from the files in `results/`; docstring scan of Rathi's 32 modules (4 missing nested-function docstrings added); folder READMEs (`src/data/` new; `src/nli/`, `src/demo/` updated); `python -m pytest -q`.
- **Settings:** every number copied from a named file in `results/` or `models/index_v1/meta.json`; status at the results freeze (Wed 18 Nov, 23:00). No training, so no W&B run.
- **Key numbers:** 0 TBD cells left; 142 tests pass, 8 skipped (GPU / parquet); GPUs used: A100 (R2, 2.6 min), L4 (R4: v1 19.3, v2 10.5, XLM-R 32.8 min), T4 (R3, R5, R6, H3–H5, Y2–Y3).
- **Not covered yet:** Arabic twins / experiments (Y4/Y5), baselines (F4) and the vision side (F1–F5) — added to §9 when their results are pushed.
- **Files:** `docs/implementation.md`, `src/{data,nli,demo}/README.md`; tag `v0.9` on main after this merge.
