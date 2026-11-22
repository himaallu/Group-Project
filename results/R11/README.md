# R11 · Repository v1.0 (Rathi)

- **What ran:** `python -m src.eval.figures` with the newly merged F3–F5 results. This adds formatted tables `tab_audit_E0` / `tab_audit_averitec` (F4, replacing the raw copy), `tab_detectors`, `tab_robustness`, `tab_leakage` and `tab_onnx` (F3/F5). `run_all.sh` gains `baselines` (F4) and `vision` (F5, needs `GENIMAGE_ROOT`) stages. README results now cover the verdict head, the baselines audit and vision. `docs/implementation.md` §9 has F1–F5. Contribution log and branches checked. Fresh-clone test of `run_all.sh`. Tag `v1.0`.
- **Settings:** no training, so no W&B run. Every number is read from files in `results/`.
- **Key numbers:** the 10 earlier R8 tables and 5 figures rebuild byte-for-byte; 6 new tables (12 files). Tests: 154 pass, 8 skipped locally; F4/F5 tests need lightgbm/torch and run in Colab.
- **Open (owners):** missing contribution lines H5, H6, H7 (Himasri) and Y1, Y2, Y3 (Youssef). Unmerged branches: PR #33 / `claude/serene-galileo-k69y8z`, `himasri/H7-qwen4b{,-swap}` (Himasri); `feba/F2-vision-data`, `feba/F5-vision-experiments` (Feba). `rathi/R10-demo-polish` is superseded by #40. Y4/Y5 Arabic results are not pushed yet: when they are, rerun `bash run_all.sh` (adds `tab_arabic_E7`).
- **Files:** `src/eval/figures.py`, `tests/eval/test_r11.py`, `run_all.sh`, `README.md`, `docs/implementation.md`, `reports/tables/tab_{audit_E0,audit_averitec,detectors,robustness,leakage,onnx}.{csv,tex}`.
- **Update, Sun 22 Nov (`v1.0.1`):**
  - Tables rebuilt after Feba's F3/F5 re-run (96 px pixel-only input) and Youssef's Y5; new `tab_onnx_int8`.
  - README vision results rewritten: AUROC 0.61–0.75; fused ALR 22.2% vs decoupled 11.1%.
  - `docs/implementation.md` §9 now covers F2, F3, F5, F6, Y4, Y5, Y6, H7, the twin hand-check and T1.
  - Tests: 166 pass, 8 skipped locally.
  - Still missing from the contribution log: H5, H6, H7 (Himasri).
