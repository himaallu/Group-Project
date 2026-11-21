# MAAI902 report: fact sheet per section (Himasri) — updated Sat 21 Nov (final experiments)

Facts and numbers only, each with the file it comes from. **Write every sentence in your own words** (Turnitin < 10%, and no paragraphs shared with the MAAI901 report or with the paper). Figures/tables already generated are marked [FIG] / [TAB]. "pp" = percentage points.

---

## 1. Introduction
- Problem: fact-checkers reach good scores by shortcuts (claim–evidence similarity, claim plausibility) instead of reading evidence — Papadopoulos et al., WACV 2025 ("Similarity over Factuality").
- Arabic / mixed Arabic–English content is rarely tested, yet common in UAE forwarded messages.
- Image side (one sentence; details belong in MAAI901): an AI-looking photo can push a true post to "fake" (FALSDO, IJCAI 2026).
- Question: does the verdict change for the right reason when evidence changes, and stay the same when only wording / language changes?
- Contributions (say "to our knowledge"): (1) twins — controlled one-change copies of real claims (`_P _A _M _B _C`); (2) twin-loss training for an evidence-based verdict head; (3) English + Arabic + mixed evaluation; (4) evidence that accuracy and evidence-reading diverge: the most accurate models (claim-only, fine-tuned Qwen3-4B) fail the twin tests.
- Headline numbers to preview: twin loss ESR 52 → 90%, unrelated 42 → 100% for ~1 F1 point (`results/H4/README.md`); claim-only (no evidence) 0.481 macro-F1 vs our 0.352 (`results/F4/audit_E0.csv`).

## 2. Related work
- VitaminC (Schuster et al., NAACL 2021): contrastive evidence edits, clean English Wikipedia only.
- CCN (Abdelnabi et al., CVPR 2022): real news + web evidence, but only with vs without evidence.
- FakeWorld (ICML 2026): unifies authenticity and truth — not our novelty; our design rule is the opposite (verdict never sees the authenticity score).
- Datasets: MultiFC (Augenstein et al., 2019), AVeriTeC, X-FACT, M4FC.
- NLI for fact verification; calibration (temperature scaling); LLMs as judges and for claim extraction; QLoRA for quantized fine-tuning.
- TODO before writing: FALSDO and FakeWorld entries in `references.bib` still have placeholder authors.

## 3. Data and twins
- MultiFC: authors' release with snippet text; own 80/10/10 split, stratified, seed 42 (official test labels are dummies) → train 16,847 / val 2,104 / test 2,106 claims (`results/R1/counts.csv`).
- Label map: 116 MultiFC labels → supported / contradicted / unverified; 9,877 claims dropped (half true, misleading, satire, …) (`notes/R1.md`).
- Cleaning (MultiFC): 197,258 → 154,277 snippets; fact-check-site snippets −25,993; claim repeats −16,769; 426 claims with no snippet set aside (`notes/R1.md`).
- Class balance: unverified ~3.5% of MultiFC → macro-F1 + class-weighted loss.
- AVeriTeC: 2,309 / 252 / 429 claims; QA evidence as "question? answer" snippets (`results/R1/counts.csv`).
- Arabic: X-FACT Arabic (from misbar.com; 135 test claims, 91% contradicted) and M4FC Arabic (test only, no evidence text) (`notes/Y1.md`, `results/Y5/README.md`).
- English twins (R3): 500 test + 3,000 train sources; test 955 twins (174 `_P`, 281 `_B`, 500 `_C`), train 5,705; automatic checks 0 failures in test (`results/R3/README.md`).
- Arabic / mixed twins (Y4): 200 test sources → 642 twins (147 `_A`, 192 `_M`, 145 `_B_A`, 158 `_C_A`), Qwen3-8B translation with a 3,044-name glossary (`results/Y4/README.md`). Hand check of 50 translations: **27/50 (54%) fully correct** (`results/Y4/check_50.csv`).
- Twin hand-check (Himasri + Feba, 120 test twins, 30 shared): **kappa 0.57**, raw agreement 83%; valid rate `_P` 97.5%, `_C` 100%, **`_B` 22.5%** (`results/H4/handcheck_summary.json`). All 5 disagreements are `_B` (Feba judged entity type, Himasri judged whether the swap contradicts the claim). Main `_B` problems: nonsense entities, surname-only swaps, original entity still present, swap not in the claim, original evidence never addressed the claim (`notes/H4.md`).
- Paraphrases are conservative (~40% near copies) (`notes/H4.md`).
- Gold NLI pairs: 150 EN + 150 AR, hand-labelled (`data/nli/gold_val.jsonl`).

## 4. Method
- Pipeline (`src/pipeline.py`, all six steps real): claim extraction (Qwen few-shot) → offline retrieval (BM25 + BGE-M3, top 20 → reranker → top 5) → relevance (fine-tuned `bge-reranker-v2-m3`) + entities (spaCy) → NLI (mDeBERTa-v3 XNLI, zero-shot) → verdict head → calibration → cited explanation.
- Verdict head: MLP 10 → 64 → 3, dropout 0.2; 10 features: relevance-weighted entail / contradict / neutral, max entail / contradict over relevant snippets, share relevant, entity match place / date / person, number of snippets (`src/verdict/{model,features}.py`).
- Twin loss: class-weighted CE on originals + λ_inv · KL(p_orig ‖ p_twin) for `_P/_A/_M` (original detached) + λ_sens · CE(`_B` → contradicted, `_C` → unverified); grouped batches (`src/verdict/train.py`).
- Calibration: temperature T by val NLL, then Unverified threshold τ by val macro-F1 (`src/verdict/calibrate.py`).
- LLM alternative (H7): Qwen3-4B 4-bit + LoRA (QLoRA) reads claim + top-3 snippets and outputs the label; trained on originals only or originals + twins (`src/verdict/llm_verdict.py`).
- Design rule: the verdict head never sees the image-authenticity score.

## 5. Experimental setup
- Adam lr 1e-3, ≤ 30 epochs, early stopping on val macro-F1 (patience 5), 64 groups per batch; seeds 42/43/44, mean ± sd.
- λ grid λ_inv ∈ {0, 0.5, 1, 2} × λ_sens ∈ {0, 0.5, 1}; best val twin score within 1 F1 point of λ = 0 → λ_inv 1, λ_sens 0.5 (`results/H4/selected_lambda.json`) [FIG `fig_lambda_grid.png`].
- Metrics (`src/eval/metrics.py`): macro-F1, AUROC, ECE (15 bins); ESR, unrelated rate, paraphrase and cross-lingual consistency; paired bootstrap 1,000 resamples by base id (`src/eval/compare.py`).
- Test sets: MultiFC test + twins; AVeriTeC test (429); X-FACT Arabic test (135); Arabic/mixed twin set (192 sources).
- Baselines (F4): claim-only TF-IDF + LR; similarity features + LightGBM; zero-shot NLI rule; LLM judge (Qwen3-8B, and Qwen3-4B in H7).
- Ablations: no twin loss, no reranker, no entity match, zero-shot vs fine-tuned NLI.
- Model selection (H7): verdict-head classifiers; NLI checkpoints on gold; Qwen3-4B vs 8B in every LLM role; Qwen3-4B as the verdict model.
- Compute: free Colab T4 for most runs; L4 for R4 / F5; QLoRA 66 / 93 min on a T4 (`results/H7/qwen4b/metrics_llm4b_*.json`).
- Human evaluation T1: 60 explanation notes (40 EN rated by Himasri and Feba, 20 AR by Youssef): verdict correct, sentences supported, readability 1–5.

## 6. Results
- Main tables [TAB `reports/tables/main_results.tex`, `twin_results.tex`, R8 `tab_*`].
- **Twin loss (MultiFC test, 3 seeds):** ESR 52.2 → 89.7; unrelated 42.4 → 100; paraphrase 91.2 → 91.4 (n.s.); macro-F1 36.3 → 35.2; ECE 5.4 → 8.2 (7.0 calibrated). Bootstrap: ESR +35 to +42, unrelated +32 to +81 pp, p < 0.001 every seed (`results/H4/`).
- **Baselines on MultiFC** (`results/F4/audit_E0.csv`): claim-only 0.481 F1 / ESR 0.445 / unrelated 0.012; similarity 0.385 / 0.592 / 0.014; zero-shot NLI rule 0.346 / 0.623 / 0.536; LLM judge 8B 0.233 / 0.100 / 0.950. Always-contradicted 0.257.
- **Ablations** [FIG `fig_ablations.png`] (`results/H5/summary.csv`): no reranker → F1 30.1, ESR 68.7, unrelated 81.0 (p < 0.001); no entity match → 33.8 / 86.8 (significant in one seed).
- **AVeriTeC:** twin 38.3 vs notwin 38.0 macro-F1 (n.s.); LLM judge 4B 0.613 is the best AVeriTeC score in the project; zero-shot NLI rule 0.440, 8B judge 0.494 (`results/H5/`, `results/F4/`, `results/H7/qwen4b/`).
- **Calibration transfer:** AVeriTeC ECE 9.8 → 5.0 (notwin); X-FACT Arabic ECE ~0.3, English calibration helps a little, Arabic-fitted 0.28 → 0.18 (notwin) (`results/H5/e5_xfact_ar_summary.csv`).
- **Model selection — verdict head** (`results/H7/verdict_head_selection.csv`): LightGBM best F1 without twins (38.4) but paraphrase 66.7, unrelated 7.4; twins as data help all (ESR ~80); MLP + twin loss +8.5 to +9.3 ESR over twins-as-data (p < 0.001).
- **Model selection — NLI on gold** (`results/H7/nli_selection.csv`): mDeBERTa-base EN 0.367 / AR 0.443 (279M, 90 ms); XLM-R-large 0.382 / 0.502 (560M); DeBERTa-v3-large EN 0.427 (EN only); fine-tuned v1 0.124 / 0.168, v2 0.334 / 0.473. CIs overlap.
- **Qwen3-4B vs 8B** (`results/H7/qwen4b/README.md`): claim extraction F1 0.977 vs 0.984; explanation faithfulness 0.352 vs 0.320, citation accuracy 0.363 vs 0.306; LLM judge MultiFC 0.391 vs 0.233, AVeriTeC 0.613 vs 0.494 (p < 0.001).
- **Qwen3-4B fine-tuned as the verdict model** (seed 42, `bootstrap_llm4b.json`): originals → **F1 0.500** (best), ESR 0.306, unrelated 0.148, paraphrase 0.787; + twins → F1 0.442, ESR 0.673, unrelated 0.992, paraphrase 0.874 (twins: ESR +36.7, unrelated +84.4 pp, p < 0.001); MLP twin loss still higher ESR (+21.7, p < 0.001).
- **Arabic / mixed twins (E7, `results/Y5/README.md`)**: fair system twin_noent — Arabic F1 0.597 vs English 0.605, unrelated 100% in both, cross-lingual consistency 73.9% (`_A`) / 79.5% (`_M`); twin (with entity features) 63.0% / 77.3%; notwin 62.4% (`_A`). X-FACT Arabic: all systems ≈ chance (0.19–0.33).
- **Components:** retrieval recall@5 EN 0.945 / AR 0.92; reranker AUROC 0.980 → 0.998; explanation faithfulness 0.32; claim extraction 0.984 vs 0.786 rule (`results/R2`, `R5`, `Y3`).
- **Human evaluation T1** (`results/T1/human_eval_summary.json`): EN agreement Himasri–Feba kappa verdict 0.49, sentences 0.35, readability 0.48 (weighted); verdict judged correct from the snippets 37.5% (Himasri) / 47.5% (Feba) EN, 10% AR (Youssef); fully supported notes 35% / 55% EN, 25% AR; readability 3.9 / 4.1 EN, 3.7 AR.

## 7. Analysis and limitations
- **Accuracy and evidence-reading diverge:** claim-only (no evidence) 0.481 and fine-tuned Qwen3-4B 0.500 are the most accurate, yet fail the twins (unrelated 0.01 / 0.15); twin training moves every model toward evidence-reading at a small accuracy cost.
- **The similarity shortcut, measured:** zero-shot max entail 0.594 (true) vs 0.579 (false); 34% of false claims have a snippet with entail > 0.9 (`results/H5/zeroshot_signal_by_label.json`) [FIG `fig_entail_shortcut.png`]; Youssef finds the same on X-FACT Arabic (15 of 36 failures are false rumours repeated by snippets, `notes/Y6.md`).
- **MultiFC snippets often do not settle the claim:** T1 raters judge the gold verdict supported by the cited snippets in only 38–48% of EN notes (10% AR).
- **NLI fine-tuning learned the shortcut:** v1 gold 0.371 → 0.124 (neutral 1/132); v2 tie; v2 per-pair: 71% of new "contradict" errors mention an entity not in the claim vs 45% baseline (n = 14).
- **ESR caveats:** (i) twin heads lean to "contradicted" (supported recall 20% EN); paired ESR (only sources not already called contradicted) is 76.4% twin vs 32.7% notwin EN (`results/Y5/diagnostics.csv`); (ii) only 22.5% of `_B` twins are valid by hand-check.
- **Unrelated = 100% is partly easy:** `_C` evidence has reranker relevance < 0.1.
- **Arabic gap:** ~85% of Arabic errors are also English errors; of the rest, 12/20 are translation problems (`results/Y6/`); only 54% of checked translations fully correct; entity features are English-only (twin vs twin_noent).
- **LLM judges fail in opposite directions:** 8B rarely says contradicted (recall 0.06), 4B rarely unverified (0.01).
- **Calibration:** T < 1 (under-confident head); τ stays 0 because the head already over-predicts unverified.
- **Limits:** MultiFC licence (research use); tiny unverified class; seed 42 only for QLoRA; small Arabic sets; claim-extraction messages template-based; X-FACT Arabic has mojibake / dummy evidence (12 test claims).
- Error analysis: 10 cases from `results/H6/error_cases.csv` (fill the `cause` column).

## 8. Conclusion
- Twins expose what accuracy hides; twin training makes models follow evidence (MLP ESR +38 pp; Qwen3-4B ESR +37 pp, unrelated +84 pp) at a small accuracy cost.
- A small, interpretable head with the twin loss reads evidence best; a quantized 4B LLM is the most accurate but needs twins to stop guessing from the claim.
- The evidence itself (search snippets) and the twin generator are the main limits.
- Future work: human-labelled on-topic NLI pairs; better `_B` swaps (claim-entity only, validated); stronger paraphrases; multilingual entity matching; better Arabic translation checks; real forwarded messages and dialects.

## Also required
- Team contribution statement from `notes/contributions.md`.
- Code and data availability: GitHub repo (tag v1.0); MultiFC research-use licence; models on Hugging Face (`RathiAbeth/twincheck-reranker-v1`).
