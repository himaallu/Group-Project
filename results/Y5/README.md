# Y5 · E7 Arabic and mixed-language evaluation (Youssef)

Code: `src/eval/run_y5_ar.py` (E7 table), `src/eval/y5_diagnostics.py` (bias checks).
Inputs: Y4 twins, zero-shot mDeBERTa NLI (`features/nli_zeroshot/copies_ar/`), fine-tuned reranker
(`features/relevance_copies_ar.parquet`), H5 verdict heads, F4 baselines (Arabic runs in `baselines/`).
Reproduced on CPU from the repo files: identical numbers.

## Twin set (192 sources): mean over seeds 42/43/44 (sd in `arabic_E7.csv`)
Language groups: **en** = English original + `_B` + `_C` (576 records); **ar** = `_A` + `_B_A` + `_C_A`
(450); **mixed** = `_M` (192, all supported, so no macro-F1).

| System | Lang | Macro-F1 | ESR | Unrelated | Cross-lingual |
|---|---|---|---|---|---|
| twin | en | 0.643 | 89.1 | 100 | |
| twin | ar | 0.559 | 96.8 | 100 | 63.0 (`_A`) |
| twin | mixed | | | | 77.3 (`_M`) |
| **twin_noent** | en | 0.605 | 87.2 | 100 | |
| **twin_noent** | ar | **0.597** | 80.5 | 100 | **73.9** (`_A`) |
| twin_noent | mixed | | | | 79.5 (`_M`) |
| notwin | en | 0.356 | 51.7 | 23.1 | |
| notwin | ar | 0.390 | 39.5 | 24.7 | 62.4 |
| zero-shot NLI rule | en | 0.541 | 60.4 | 54.7 | |
| zero-shot NLI rule | ar | 0.494 | 52.4 | 45.6 | 64.6 |
| similarity + LightGBM | en | 0.290 | 57.8 | 2.1 | |
| similarity + LightGBM | ar | 0.235 | 74.3 | 0.6 | 49.4 |

Cross-lingual = % of `_A` / `_M` twins with the same verdict as the English original.

## X-FACT Arabic test (135 real claims with evidence)
twin 0.328, twin_noent 0.326, notwin 0.320, similarity 0.312, zero-shot NLI 0.193 macro-F1.
Same as E5: all close to chance on this small, 91%-false set (failures in `results/Y6/`).

## Read the ESR with `diagnostics.csv`
The twin heads lean towards "contradicted" (on English originals too). With the entity features
(filled with 1.0 for Arabic, since spaCy is English-only) this gets much worse in Arabic:

| System | Lang | Supported recall | Paired ESR (n) | Arabic errors also wrong in English |
|---|---|---|---|---|
| twin | en | 20.3 | 76.4 (75) | |
| twin | ar | **1.8** | 62.9 (6) | 85.0 |
| twin_noent | en | 14.1 | 58.5 (47) | |
| twin_noent | ar | 15.6 | 40.5 (33) | 85.7 |
| notwin | en | 37.3 | 32.7 (116) | |
| notwin | ar | 59.4 | 16.4 (90) | 85.3 |

- Paired ESR counts only sources whose pre-swap twin was *not* already called contradicted, so it
  measures a real verdict change. twin's 96.8% Arabic ESR is mostly bias (only 6 eligible sources).
- **twin_noent is the fair cross-language system**: Arabic macro-F1 0.597 vs English 0.605, unrelated
  detection 100% in both, supported recall 15.6% vs 14.1%. Arabic loses on evidence sensitivity
  (paired ESR 40.5 vs 58.5), the main language gap.
- About 85% of wrong Arabic or mixed verdicts are also wrong in English, so most errors come from the
  head, not from Arabic. The other 15% are studied in Y6.
- The similarity baseline's TF-IDF vocabulary is English-only, so Arabic falls back on token overlap.
- M4FC Arabic is not in E7: it has no evidence text.

Files: `arabic_E7.csv`, `arabic_E7_by_seed.csv`, `diagnostics.csv`, `preds_<system>_s<seed>.jsonl`,
`baselines/`.
