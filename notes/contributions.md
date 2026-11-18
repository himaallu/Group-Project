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
