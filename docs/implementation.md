# Implementation details

_Owner: Rathi (R9). For each model and dataset: checkpoint, hyperparameters, training
time, GPU, data sizes and twin counts. Every number here is copied from a file in
`results/` (named next to it). Status at the results freeze (Wed 18 Nov 2026, 23:00);
Arabic experiments (Y4/Y5), baselines (F4) and the vision side (F1–F5) are added when
their results are pushed._

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.11 locally; Colab's Python (3.12/3.13) for GPU jobs |
| Libraries | `requirements.txt` (torch, transformers, sentence-transformers, spaCy, rank_bm25, faiss-cpu, scikit-learn, wandb, gradio, …) |
| GPU | Colab Pro: A100-SXM4-80GB (R2), L4 (R4), T4 (R3, R5, R6, H3–H5, Y2–Y3); the verdict head trains on CPU |
| Tracking | Weights & Biases, project `twincheck`, seed 42 (43, 44 for repeats) |
| Large files | Hugging Face Hub (see §7); everything else in git |
| Tests | `python -m pytest -q` (10-row fixtures; no GPU or downloads) |

## 2. Data (R1, Y1)

Source: `results/R1/counts.csv`, `notes/R1.md`, `notes/Y1.md`.

| Dataset | Train | Val | Test | Snippets per claim | Notes |
|---|---|---|---|---|---|
| MultiFC | 16,847 | 2,104 | 2,106 | 7.3 | own 80/10/10 split, stratified, seed 42 (official test labels never released) |
| AVeriTeC | 2,309 | 252 | 429 | ~2.5 | official dev = our test; each QA answer = one snippet |
| X-FACT Arabic | 670 | 93 | 135 | 3.4 | authors' splits (dev = val), after `clean.py`; 21 claims with no snippet kept apart (Y1) |
| M4FC Arabic | – | – | 769 | 0 (image + caption) | test only (Y1) |

Cleaning (`src/data/clean.py`), MultiFC: 197,258 → 154,277 snippets (fact-check sites
−25,993; claim repeats −16,769; duplicates −219 snippets and −73 claims; 426 claims with
no snippet kept apart). Label map: `data/label_map.csv` (9,877 MultiFC claims dropped as
half true / misleading / satire / undocumented scores).

## 3. Reranker (R2)

Source: `results/R2/reranker_eval.json`, `notes/R2.md`.

| Item | Value |
|---|---|
| Base checkpoint | `BAAI/bge-reranker-v2-m3` (cross-encoder, one logit; relevance = sigmoid) |
| Training data | 20,000 pairs from MultiFC + AVeriTeC train (own snippet = 1, random other claim's snippet = 0), seed 42 |
| Hyperparameters | 1 epoch, AdamW lr 2e-5, weight decay 0.01, batch 16, max length 256, 10% linear warmup, fp16, grad clip 1.0 |
| Training time / GPU | 2.6 min, A100-SXM4-80GB |
| Validation (2,000 pairs) | accuracy 0.719 → 0.980; AUROC 0.980 → 0.998 |
| Output | `features/relevance_en.parquet` (`id, snippet_idx, relevance`); model `RathiAbeth/twincheck-reranker-v1` |
| W&B | run `aihv3ag6` (reported); Hub weights from run `yftu2n5c` (same settings) |

Entities (`src/data/entities.py`): spaCy `en_core_web_trf` (falls back to
`en_core_web_lg` on Colab's newer Python), labels PERSON, GPE, LOC, ORG, DATE, EVENT with
character offsets → `features/entities_en.jsonl` (24,047 claims, 161,375 snippets).

## 4. Twins (R3)

Source: `results/R3/make_twins_report.json`, `results/R3/checks.json`, `notes/R3.md`.

| Split | Sources | `_P` | `_B` | `_C` | Check failures |
|---|---|---|---|---|---|
| test | 500 | 174 | 281 | 500 | 0 |
| train | 3,000 | 986 | 1,719 | 3,000 | 1 (rounding of a logged score) |

- Sources: supported claims with ≥ 2 snippets of relevance > 0.3, seed 42 (all MultiFC).
- `_B`: one GPE / person (full name and surname) / 4-digit year swapped in every snippet;
  replacement names from train-split entities seen ≥ 5 times; years ±2–5, never after
  2025. Swap kinds: 904 place, 898 person, 198 year.
- `_C`: donor evidence with no shared entity and every snippet below reranker relevance 0.1 (≤ 20 tries).
- `_P`: `Qwen/Qwen3-8B`, 4-bit nf4, thinking off, greedy first try then sampling, ≤ 96
  new tokens, up to 3 tries; kept only if spaCy entities and numbers match (35% kept).

## 5. NLI (R4)

Source: `results/R4/{v1_,}gold_eval.json`, `results/R4/v1_gold_eval_xlmr.json`,
`results/R4/{v1_,}pairs_report.json`, `notes/R4.md`.

| Item | Value |
|---|---|
| Base checkpoint | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`; comparison `joeddav/xlm-roberta-large-xnli` |
| v1 training data | 32,658 balanced weak pairs from train splits only; label copied from the claim (supported claim + relevant snippet → entail, contradicted claim + relevant snippet or `_B` swapped snippet → contradict, random other claim's snippet → neutral; ≤ 2 per claim, relevance ≥ 0.3) + 2,658 Arabic pairs (Y2) |
| v2 training data | 18,525 pairs (6,175 per label; 17,880 EN + 645 AR): a weak pair is kept only if the zero-shot model gives its label as the top class with p ≥ 0.5; plus "hard neutrals" (a claim + its own snippet that the zero-shot model calls neutral); 3,226 `_B` pairs kept unfiltered |
| Input | premise = snippet, hypothesis = claim; max length 256, `longest_first` truncation |
| Hyperparameters | 2 epochs, AdamW lr 2e-5 (XLM-R: 1e-5), batch 16, 10% linear warmup, fp32 with TF32 (the checkpoint is stored in fp16 and cast to fp32), seed 42 |
| Training time / GPU | NVIDIA L4: v1 19.3 min, v2 10.5 min, XLM-R v1 32.8 min; 0 skipped batches |
| Gold macro-F1 (150 EN / 150 AR) | mDeBERTa zero-shot 0.371 / 0.443; v1 0.124 / 0.162 (label shortcut); v2 0.334 / 0.473 (tie with zero-shot, CIs overlap); XLM-R zero-shot 0.382 / 0.502, v1 0.124 / 0.174 |
| Decision | zero-shot mDeBERTa stays the default (`src/nli/score.py`); v2 scores kept as an ablation (`features/nli_v2/`) |
| W&B | v2 `hmtd5olh`, v1 `gepd51e6`, XLM-R v1 `e50h261l` (project `twincheck`) |
| Output | `features/nli_{v1,v2}/<file>.parquet` (`id, snippet_idx, p_entail, p_contradict, p_neutral`); models `RathiAbeth/twincheck-nli-v1`, `-v2` |

Why v2 does not beat zero-shot (from the confusion matrices): the filter keeps only
pairs the zero-shot model already gets right, so v2 cannot learn its mistakes; the 66
English gold neutrals that zero-shot mislabels are the same kind v2 misses; training is
33% neutral while gold is 88% (EN) / 61% (AR) neutral; and the gold set has only 18
non-neutral English pairs, so a few pairs move macro-F1 by several points.

## 6. Explanations and offline index (R5)

Source: `results/R5/explain_eval.json`, `results/R5/retrieval_eval.json`, `models/index_v1/meta.json`.

| Item | Value |
|---|---|
| Note generator | `Qwen/Qwen3-8B`, 4-bit nf4, greedy, ≤ 160 new tokens; top 3 snippets by reranker; 1–2 sentences in the claim's language; citations outside [1]–[3] removed. Verdict explained = gold label (250 test claims: 200 EN, 50 AR, seed 42) |
| Faithfulness judge | zero-shot mDeBERTa NLI: a sentence is faithful if a cited snippet entails it with p > 0.5 |
| Faithfulness / citation accuracy | all 0.320 / 0.306 (EN 0.287 / 0.276, AR 0.484 / 0.448); notes fully faithful 20.8% (EN 16%, AR 40%) |
| Library | 163,244 unique snippets from 24,945 claims (MultiFC, AVeriTeC, X-FACT AR, all splits) |
| Search | BM25 top 20 ∪ BGE-M3 (`BAAI/bge-m3`, FAISS inner product) top 20 → fine-tuned reranker → best 5 |
| Recall@5 (BM25 / dense / hybrid) | EN 0.91 / 0.87 / 0.945; AR 0.86 / 0.80 / 0.92 |
| Output | library `RathiAbeth/twincheck-index-v1` (dataset); `src/retrieval/index.search(claim, k)` |

## 7. Large files on the Hugging Face Hub

GitHub's free LFS quota (1 GB for the repo) is too small, so (agreed deviation from CLAUDE.md §6):

| Repo | Contents | Size |
|---|---|---|
| `RathiAbeth/twincheck-reranker-v1` | fine-tuned reranker (fp16 safetensors) | ~1.1 GB |
| `RathiAbeth/twincheck-nli-v1`, `-v2` | fine-tuned mDeBERTa (safetensors) | ~1.1 GB each |
| `RathiAbeth/twincheck-index-v1` (dataset) | snippets + BGE-M3 embeddings (163,244 × 1,024, fp16) | ~330 MB |

## 8. Demo (R6)

Source: `results/R6/README.md`, `results/R6/examples{,_rule}.json`.

`src/demo/app.py` (Gradio) calls `src/pipeline.check`; six made-up UAE examples
(`src/demo/examples.py`, 2 per verdict, 2 Arabic, 1 mixed) are shown as
"Illustrative example". Every component real on a Colab T4; the extractor and the
explainer share one 4-bit Qwen3-8B (two copies plus reranker and spaCy do not fit in
15 GB). With the trained verdict head (`models/verdict_twin_s42`) 2 of 6 examples get
the expected verdict; with the rule stand-in (max entail vs max contradict,
`TWINCHECK_VERDICT=none`) 5 of 6.

## 9. Teammates' components

Copied from each owner's `results/<task-id>/README.md` (details there).

| Component | Owner | Settings | Key numbers |
|---|---|---|---|
| Arabic NLI pairs | Youssef (Y2) | X-FACT AR weak pairs + MultiFC pairs translated with Qwen3-8B 4-bit (greedy); 12% of translations dropped (wrong script / left in English) | 2,658 balanced pairs; 150 Arabic gold pairs (36 e / 22 c / 92 n) |
| Claim extraction | Youssef (Y3) | Qwen3-8B 4-bit few-shot (6 examples, 2 per language), greedy; match = BGE-M3 cosine ≥ 0.8 | F1 on 200 held-out messages: 0.984 (EN 0.989, AR 0.966, mixed 1.000) vs rule split 0.786 |
| English gold NLI pairs | Himasri (H2) | 150 MultiFC val pairs, 50 per verdict label, hand-labelled | 10 entail / 8 contradict / 132 neutral |
| Verdict head | Himasri (H3) | MLP 10 → 64 → 3 (dropout 0.2) on zero-shot NLI + relevance + entity features; class-weighted CE; Adam lr 1e-3, batch 64 groups, ≤ 30 epochs, patience 5; CPU | v0 test macro-F1 0.363 ± 0.005, AUROC 0.611 ± 0.003 |
| Twin loss + calibration | Himasri (H4) | grid λ_inv ∈ {0, 0.5, 1, 2} × λ_sens ∈ {0, 0.5, 1}, chosen λ_inv = 1, λ_sens = 0.5; temperature scaling (T 0.80–0.89, τ = 0) | ESR 52.2 → 89.7, unrelated 42.4 → 100.0, macro-F1 36.3 → 35.2 (%, 3 seeds); bootstrap p < 0.001 for ESR and unrelated |
| Ablations, AVeriTeC, Arabic | Himasri (H5) | 4 variants × seeds 42/43/44; paired bootstrap 1,000 resamples | no reranker: macro-F1 30.1, ESR 68.7; AVeriTeC macro-F1 38.3 (twin) vs 38.0; X-FACT AR near chance (AUROC 0.51–0.54) |

All tables and figures in `reports/` are rebuilt from these files by `python -m src.eval.figures` (R8).
