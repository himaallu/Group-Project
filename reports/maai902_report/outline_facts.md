# MAAI902 report: fact sheet per section (Himasri)

Facts and numbers only, each with the file it comes from. **Write every sentence in your own words** (Turnitin < 10%, and no paragraphs shared with the MAAI901 report or with the paper). Figures/tables already generated are marked [FIG] / [TAB].

---

## 1. Introduction
- Problem: fact-checkers reach good scores by shortcuts (claim–evidence similarity, claim plausibility) instead of reading evidence — Papadopoulos et al., WACV 2025 ("Similarity over Factuality").
- Arabic / mixed Arabic–English content is rarely tested, yet common in UAE forwarded messages.
- Image side (one sentence; details belong in MAAI901): an AI-looking photo can push a true post to "fake" (FALSDO, IJCAI 2026).
- Our question: does the verdict change for the right reason when evidence changes, and stay the same when only wording / language changes?
- Contributions (say "to our knowledge"): (1) twins — controlled one-change copies of real claims (`_P _A _M _B _C`); (2) twin-loss training for an evidence-based verdict head; (3) English + Arabic + mixed evaluation; (4) an analysis showing the similarity shortcut inside zero-shot and fine-tuned NLI.
- Headline numbers to preview: ESR 52 → 90%, unrelated 42 → 100% at ~1 F1 point cost (`results/H4/README.md`).

## 2. Related work
- VitaminC (Schuster et al., NAACL 2021): contrastive evidence edits, clean English Wikipedia only.
- CCN (Abdelnabi et al., CVPR 2022): real news + web evidence, but only with vs without evidence.
- FakeWorld (ICML 2026): unifies authenticity and truth — not our novelty; our design rule is the opposite (verdict never sees the authenticity score).
- Datasets as related work: MultiFC (Augenstein et al., 2019), AVeriTeC, X-FACT, M4FC.
- NLI for fact verification; calibration (temperature scaling); claim extraction with LLMs.
- TODO before writing: FALSDO and FakeWorld entries in `references.bib` still have placeholder authors.

## 3. Data and twins
- MultiFC: authors' release with snippet text (HF mirror and Codalab have no snippets); own 80/10/10 split, stratified, seed 42 (official test labels are dummies) → train 16,847 / val 2,104 / test 2,106 claims (`results/R1/counts.csv`).
- Label map: 116 MultiFC labels mapped to supported / contradicted / unverified; 9,877 claims dropped (half true, misleading, satire, …) (`notes/R1.md`).
- Cleaning (MultiFC): 197,258 → 154,277 snippets; fact-check-site snippets −25,993 (politifact 13,585, snopes 4,002, factcheck.org 1,847); claim repeats −16,769; 426 claims left with no snippet (`notes/R1.md`).
- Class balance: unverified only ~3.5% of MultiFC → macro-F1 + class-weighted loss (`notes/R1.md`).
- AVeriTeC: 2,309 / 252 / 429 claims; QA evidence turned into "question? answer" snippets (`results/R1/counts.csv`, `notes/R1.md`).
- Arabic: X-FACT Arabic (all from misbar.com; 838 contradicted / 55 supported / 26 unverified before cleaning; ~25% of evidence links are fact-check sites) and M4FC Arabic (614 claims, test only) (`notes/Y1.md`).
- Twins (R3): sources = supported claims with ≥ 2 relevant snippets; 500 test + 3,000 train sources; test 955 twins (174 `_P`, 281 `_B`, 500 `_C`), train 5,705; automatic checks: 0 failures in test (`results/R3/README.md`, `results/R3/checks.json`).
- Twin quality notes: paraphrases are conservative — ~40% have word overlap ≥ 0.9 with the original, only 6% are real rewordings (my check, see H4 discussion); `_B` impossible for 44% of sources (no claim entity in snippets) (`results/R3/README.md`).
- Gold NLI pairs: 150 EN (10 entail / 8 contradict / 132 neutral) + 150 AR (36 / 22 / 92), hand-labelled (`results/H2/README.md`, `data/nli/gold_val.jsonl`).
- Twin hand-check: 120 test twins, 30 rated by both Himasri and Feba → Cohen's kappa (`results/H4/handcheck_summary.json`, pending).

## 4. Method
- Pipeline (`src/pipeline.py`): claim extraction (Qwen3-8B few-shot) → offline retrieval (BM25 + BGE-M3 FAISS, top 20 → reranker → top 5) → relevance (fine-tuned `bge-reranker-v2-m3`) + entities (spaCy) → NLI (mDeBERTa-v3 XNLI, zero-shot) → verdict head → calibration → cited explanation (Qwen3-8B, cites [1]–[3]).
- Verdict head: MLP 10 → 64 → 3, dropout 0.2 (`src/verdict/model.py`). 10 features: relevance-weighted entail / contradict / neutral, max entail, max contradict over relevant snippets, share of relevant snippets (relevance ≥ 0.5), entity match for place / date / person, number of snippets (`src/verdict/features.py`).
- Twin loss (`src/verdict/train.py`): class-weighted CE on originals + λ_inv · KL(p_orig ‖ p_twin) for `_P/_A/_M` (original detached = teacher) + λ_sens · CE(`_B` → contradicted, `_C` → unverified); batches keep each original with its twins.
- Calibration (`src/verdict/calibrate.py`): temperature T by validation NLL, then "Unverified" threshold τ by validation macro-F1.
- Design rule: the verdict head never sees the image-authenticity score.

## 5. Experimental setup
- Adam lr 1e-3, ≤ 30 epochs, early stopping on val macro-F1 (patience 5), 64 groups per batch; seeds 42 / 43 / 44, mean ± sd (`results/H3/metrics_*.json`).
- λ grid: λ_inv ∈ {0, 0.5, 1, 2} × λ_sens ∈ {0, 0.5, 1}; choose best mean of val paraphrase consistency / ESR / unrelated rate with val macro-F1 ≥ λ0 − 1 point; validation twins = twins of 10% of train sources held out → λ_inv 1, λ_sens 0.5 (`results/H4/selected_lambda.json`, `results/H4/grid_summary.csv`) [FIG `reports/figures/fig_lambda_grid.png`].
- Metrics (`src/eval/metrics.py`): macro-F1, AUROC, ECE (15 bins) on originals; ESR, unrelated rate, paraphrase and cross-lingual consistency on twins; paired bootstrap 1,000 resamples by base id, 95% CI + p (`src/eval/compare.py`).
- Test sets: MultiFC test + test twins; AVeriTeC test (429); X-FACT Arabic test (135).
- Ablations: no twin loss, no reranker, no entity match, zero-shot vs fine-tuned NLI (`src/eval/run_h5.py`).
- Compute: free Colab T4 (verdict head trains on CPU in minutes); reranker fine-tuned 1 epoch on 20k pairs in 2.6 min on an A100 (`results/R2/README.md`).

## 6. Results
- Main table [TAB `reports/tables/main_results.tex`, `twin_results.tex`].
- Twin loss, MultiFC test (mean ± sd): ESR 52.2 ± 3.0 → 89.7 ± 0.9; unrelated 42.4 ± 24.4 → 100.0 ± 0.0; paraphrase consistency 91.2 → 91.4 (n.s.); macro-F1 36.3 → 35.2; ECE 5.4 → 8.2, calibrated 7.0 (`results/H4/README.md`, `results/H4/bootstrap_twin_vs_notwin.json`).
- Bootstrap: ESR +35 to +42, unrelated +32 to +81 points, p < 0.001 every seed; macro-F1 −0.1 to −2.3, significant only for seed 42 (`results/H4/bootstrap_twin_vs_notwin.json`).
- Ablations [FIG `reports/figures/fig_ablations.png`]: no reranker → macro-F1 30.1, ESR 68.7, unrelated 81.0 (p < 0.001 every seed); no entity match → 33.8 / 86.8 (significant in one seed) (`results/H5/summary.csv`, `results/H5/bootstrap.json`).
- Simple baselines: always "contradicted" 25.7; random by class prior 32.8 (`results/H3/v0_s42_per_class.json`). Feba's baselines (claim-only, similarity + LightGBM, LLM judge): pending `results/F4/`.
- AVeriTeC: twin 38.3 ± 2.0 vs notwin 38.0 ± 2.9 macro-F1 (n.s.); calibration fitted on MultiFC: ECE 9.8 → 5.0 (notwin), 11.0 → 9.0 (twin) (`results/H5/README.md`).
- X-FACT Arabic (E5): AUROC ~0.52, ECE ~0.3; English calibration helps a little, Arabic-fitted helps notwin (0.28 → 0.18) (`results/H5/e5_xfact_ar_summary.csv`).
- Components: retrieval recall@5 EN 0.945 / AR 0.92; explanation faithfulness 0.32 (EN 0.29, AR 0.48), citation accuracy 0.31 (`results/R5/README.md`); claim-extraction F1 0.984 vs 0.786 rule-based (`results/Y3/README.md`).
- Pending for the final version: cross-lingual consistency (Y4 Arabic twins), Arabic results (Y5), human evaluation T1 (60 notes), twin hand-check kappa.

## 7. Analysis and limitations
- The similarity shortcut, measured: zero-shot max entail is 0.594 for true vs 0.579 for false claims; 34% of false claims have a snippet with entail > 0.9 (snippets restate the claim) (`results/H5/zeroshot_signal_by_label.json`) [FIG `reports/figures/fig_entail_shortcut.png`]. This caps macro-F1 and makes "supported" the weakest verdict.
- NLI fine-tuning learned the shortcut: v1 (labels from verdicts) gold macro-F1 EN 0.371 → 0.124, AR 0.443 → 0.162, predicts neutral for 1 / 132 EN neutral gold pairs; v2 (zero-shot agreement filter + hard neutrals) EN 0.334 / AR 0.473, a statistical tie (overlapping CIs) (`results/R4/v1_gold_eval.json`, `results/R4/` v2 files, `src/nli/gold_check.py`).
- v2 per-pair check: of 14 EN gold neutrals v2 turns into "contradict", 71% mention a person / place not in the claim vs 45% among pairs both call neutral — suggests `_B` swap pairs teach "unfamiliar entity → contradict" (n = 14, suggestive).
- ESR caveat: the twin model calls 61% of supported source claims contradicted (notwin 40%); evidence-driven flips 18.4% → 30.5% (`results/H4/esr_vs_source_bias.json`).
- Unrelated = 100% is partly easy: `_C` evidence has reranker relevance < 0.1, so "share of relevant snippets" = 0 flags it.
- Relevance saturation: 94% of reranker scores > 0.99, yet removing the reranker costs 5 F1 / 21 ESR points — it separates related from unrelated evidence, not own snippets from each other.
- Calibration: T < 1 (class weighting makes the head under-confident); τ stays 0 because the head already over-predicts unverified (386 predicted vs 76 gold for v0) (`results/H3/v0_s42_per_class.json`, `notes/H4.md`).
- Limits: MultiFC licence unknown (research use); unverified class tiny; paraphrases conservative; Arabic test sets small and skewed (X-FACT AR 91% contradicted); claim-extraction test messages are template-based (upper bound); explanation faithfulness low because notes state what is missing.
- Error analysis: 10 cases from `results/H6/error_cases.csv` (60 candidates in 6 failure groups; fill the `cause` column).

## 8. Conclusion
- Twins expose what accuracy hides; twin-loss training makes the verdict follow evidence (ESR +38 points) at ~1 F1 point.
- Relevance is the key component; NLI trained on verdict-derived labels learns the very shortcut studied.
- Future work: human-labelled on-topic neutral NLI pairs; real forwarded messages and dialect Arabic; stronger paraphrases, number swaps, video; multilingual entity matching for Arabic twins.

## Also required in the document
- Team contribution statement from `notes/contributions.md`.
- Code and data availability: GitHub repo; MultiFC research-use licence; models on Hugging Face (`RathiAbeth/twincheck-reranker-v1`).
