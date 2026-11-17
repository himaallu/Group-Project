# Implementation details

_Owner: Rathi (R9). For each model and dataset: checkpoint, hyperparameters, training
time, GPU, data sizes and twin counts. Every number here is copied from a file in
`results/` (named next to it). Cells marked **TBD** are filled in after that task's
Colab run is pushed._

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.11 locally; Colab's Python (3.12/3.13) for GPU jobs |
| Libraries | `requirements.txt` (torch, transformers, sentence-transformers, spaCy, rank_bm25, faiss-cpu, scikit-learn, wandb, gradio, …) |
| GPU | Colab Pro, NVIDIA A100-SXM4-80GB (R2: `results/R2/reranker_eval.json`); T4 enough for R3/R6 |
| Tracking | Weights & Biases, project `twincheck`, seed 42 (43, 44 for repeats) |
| Large files | Hugging Face Hub (see §7); everything else in git |
| Tests | `python -m pytest -q` (10-row fixtures; no GPU or downloads) |

## 2. Data (R1, Y1)

Source: `results/R1/counts.csv`, `notes/R1.md`, `notes/Y1.md`.

| Dataset | Train | Val | Test | Snippets per claim | Notes |
|---|---|---|---|---|---|
| MultiFC | 16,847 | 2,104 | 2,106 | 7.3 | own 80/10/10 split, stratified, seed 42 (official test labels never released) |
| AVeriTeC | 2,309 | 252 | 429 | ~2.5 | official dev = our test; each QA answer = one snippet |
| X-FACT Arabic, M4FC Arabic | see `notes/Y1.md` | | | | Youssef (Y1) |

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

Source: `results/R4/pairs_report.json`, `results/R4/gold_eval.json`, `results/R4/gold_eval_xlmr.json`.

| Item | Value |
|---|---|
| Base checkpoint | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`; comparison `joeddav/xlm-roberta-large-xnli` |
| Training data | balanced weak pairs from train splits only: entail = supported claim + relevant snippet; contradict = contradicted claim + relevant snippet, and `_B` swapped snippets; neutral = random other claim's snippet; ≤ 2 per claim; + Arabic pairs from Y2. Counts: **TBD** (`pairs_report.json`) |
| Input order | premise = snippet, hypothesis = claim; only the snippet is truncated |
| Hyperparameters | 2 epochs, AdamW lr 2e-5 (XLM-R: 1e-5), batch 16, max length 256, 10% warmup, bf16 on A100 (fp16 elsewhere), seed 42 |
| Training time / GPU | **TBD** |
| Gold (150 EN; AR when Y2 pushes it) | macro-F1 zero-shot → fine-tuned: **TBD**. Only 10 entail and 8 contradict EN gold pairs: per-class numbers in `gold_eval.json` |
| Output | `features/nli_v1/<file>.parquet` (`id, snippet_idx, p_entail, p_contradict, p_neutral`); model `RathiAbeth/twincheck-nli-v1` |

## 6. Explanations and offline index (R5)

Source: `results/R5/explain_eval.json`, `results/R5/retrieval_eval.json`.

| Item | Value |
|---|---|
| Note generator | `Qwen/Qwen3-8B`, 4-bit, greedy, ≤ 160 new tokens; top 3 snippets by reranker; 1–2 sentences in the claim's language; citations outside [1]–[3] removed |
| Faithfulness judge | fine-tuned NLI (R4): a sentence is faithful if a cited snippet entails it with p > 0.5 |
| Faithfulness / citation accuracy (200 EN + 50 AR) | **TBD** |
| Library | every cleaned snippet of MultiFC, AVeriTeC, X-FACT AR (unique texts): **TBD** snippets |
| Search | BM25 top 20 ∪ BGE-M3 (`BAAI/bge-m3`, FAISS inner product) top 20 → fine-tuned reranker → best 5 |
| Recall@5 (BM25 / dense / hybrid) | **TBD** |
| Output | library `RathiAbeth/twincheck-index-v1` (dataset); `src/retrieval/index.search(claim, k)` |

## 7. Large files on the Hugging Face Hub

GitHub's free LFS quota (1 GB for the repo) is too small, so (agreed deviation from CLAUDE.md §6):

| Repo | Contents | Size |
|---|---|---|
| `RathiAbeth/twincheck-reranker-v1` | fine-tuned reranker (fp16 safetensors) | ~1.1 GB |
| `RathiAbeth/twincheck-nli-v1` | fine-tuned mDeBERTa (fp32 safetensors) | ~1.1 GB |
| `RathiAbeth/twincheck-index-v1` (dataset) | snippets + BGE-M3 embeddings (fp16) | **TBD** |

## 8. Demo (R6)

`src/demo/app.py` (Gradio) calls `src/pipeline.check`; six made-up UAE examples
(`src/demo/examples.py`, 2 per verdict, 2 Arabic, 1 mixed) are shown as
"Illustrative example". What the model says on them: `results/R6/examples.json` (**TBD**).

## 9. Teammates' components

Verdict head and calibration (H3, H4), claim extraction (Y3), Arabic twins (Y4),
baselines (F4) and the vision side (F1–F5) are described in each owner's
`notes/<task-id>.md` and `results/<task-id>/README.md`; their settings will be copied
here for the final version (R9, Fri).
