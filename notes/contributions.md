# Contribution log

Add one line per finished task (by 23:00 every day). Used for each document's team
contribution statement.

| Task | Person | Date | Commit / W&B link |
|---|---|---|---|
| H1 | Himasri | 2026-11-16 | branch `himasri/H1-repo-metrics` (repo layout, metrics, report, tests) |
| H2 | Himasri | 2026-11-17 | branch `himasri/H2-gold-pairs`: 150 EN gold NLI pairs (10 e / 8 c / 132 n), labelling guide, MAAI902 skeleton |
| H4 (prep) | Himasri | 2026-11-17 | branch `himasri/H4-prep`: calibration, twin hand-check sheets + kappa, `pipeline.py` interface, zero-shot NLI script |
| H3 | Himasri | 2026-11-17 | branch `himasri/H3-verdict-head`: verdict-head features/model/training with twin loss, zero-shot NLI features, v0 (test macro-F1 0.363 ± 0.005, seeds 42/43/44) |
| H4 | Himasri | 2026-11-17 | branch `himasri/H4-calibration`: λ grid (chosen λ_inv 1, λ_sens 0.5), twin vs notwin seeds 42/43/44 (ESR 52 → 90, unrelated 42 → 100, p < 0.001), calibration, bootstrap script |
| H5 (zero-shot) | Himasri | 2026-11-18 | branch `himasri/H5-experiments`: ablations (no reranker −5 F1 / −21 ESR, p < 0.001), AVeriTeC test, calibration transfer, bootstrap, report tables |
| R1 | Rathi | 2026-11-17 | PR #2 + #7: `src/data/` loaders + `clean.py`, `data/label_map.csv`, `data/processed/en/` (MultiFC 21,057 + AVeriTeC 2,990 claims) |
| R2 | Rathi | 2026-11-17 | branch `rathi/R2-reranker`: reranker AUROC 0.980 → 0.998, `features/relevance_en.parquet`, `features/entities_en.jsonl`; W&B https://forge.coreweave.com/wandb/rathiabeth-uowd/twincheck/runs/aihv3ag6 |
| R3 | Rathi | 2026-11-17 | branch `rathi/R3-twins` (dc1bd8b): `data/copies/en/` test 955 twins (174 P, 281 B, 500 C), train 5,705 (986 P, 1,719 B, 3,000 C); checks 0/1 failures |
| R4 | Rathi | 2026-11-18 | branch `rathi/R4-nli`: NLI v1 (gold EN 0.37 → 0.12, AR 0.44 → 0.16: label shortcut) and v2 (filter + hard neutrals: EN 0.33, AR 0.47, tie with zero-shot, which stays default); `features/nli_v2/`; W&B https://forge.coreweave.com/wandb/rathiabeth-uowd/twincheck/runs/hmtd5olh |
| R5 | Rathi | 2026-11-18 | branch `rathi/R5-explain-index`: offline index (163k snippets, recall@5 EN 0.945 / AR 0.92), 250 cited notes, faithfulness 0.32 / citation accuracy 0.31; index on HF `datasets/RathiAbeth/twincheck-index-v1` |
| R6 | Rathi | 2026-11-18 | branch `rathi/R6-demo` (fb01911): Gradio demo on `check()`, 6 examples; trained verdict head 2/6 as expected vs rule 5/6 (`results/R6/`); Qwen3-8B shared by extractor + explainer to fit a T4 |
| R7 | Rathi | 2026-11-18 | branch `rathi/R7-cleanup`: `run_all.sh` stages for R1–R6, Y2–Y3, H3–H5 (default run: 136 tests pass, tables rebuilt incl. AVeriTeC + X-FACT AR); README with pipeline, Hub models, reproduce, demo, provisional results |
