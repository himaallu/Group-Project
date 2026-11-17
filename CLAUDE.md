# CLAUDE.md — TwinCheck (UOWD MAAI902 + MAAI901 group project)

You are helping one member of a four-person Master's team build part of **TwinCheck**.
Read this whole file before doing anything. It is the single source of truth for the
project: the goal, the shared data format, the folder layout, the rules, and every
person's tasks with times. If anything the user asks conflicts with this file, point
it out and ask before going ahead.

- Team Brief & Build Guide (full detail): https://claude.ai/artifact/4JijVqNTSiErQoJBmoLpAz
- Task tracker (status, hour-by-hour timetable): https://claude.ai/artifact/YMpxN4ybkXpicpSexoHkdN
- Times are UAE time (UTC+4). Today's plan starts **Mon 16 Nov 2026, 23:00**.

---

## 1. How you (Claude) must work

1. Ask which team member you are helping and which task ID (e.g. `R3`) if not stated.
2. **Plan first.** Write a short implementation plan: files to create or change,
   functions with their inputs and outputs, and the order of steps. **Stop and wait
   for an OK** before writing code.
3. Build one step at a time. **Test every step on 10 rows** before running on the
   full data.
4. Use the **shared record format, file names and folders in this file exactly**.
   Never change another person's files or the format without the user agreeing it
   in the team group first.
5. Every script starts with a docstring: what it does, inputs, outputs, how to run.
   Every function has a docstring. Comment any step that is not obvious.
6. Log every training run to Weights & Biases, project `twincheck`, with the seed
   (42, then 43 and 44 for repeats).
7. At the end of a task, produce:
   - the git commands (branch, commit messages starting with the task ID, push, PR text)
   - `results/<task-id>/README.md` (5 lines: what ran, settings, W&B link, key numbers, where big files are)
   - a 5-line summary for `notes/<task-id>.md`
   - one line for `notes/contributions.md`: task ID, person, date, commit or W&B link
8. Prefer simple, readable code over clever code. Markers grade code quality.
9. If a step would take much longer than the time slot in Section 10, say so early
   and suggest the smallest change that keeps the deadline.

---

## 2. The project in one page

**Problem.** Fact-checking systems compare a claim (a news caption, a forwarded
message) with evidence found online and decide if it is true. They score well, but
studies show they often get the right answer **without really reading the evidence**:
they use shortcuts like word overlap between claim and evidence, or how plausible the
claim sounds (Papadopoulos et al., *Similarity over Factuality*, WACV 2025). They are
also rarely tested on **Arabic** or mixed Arabic–English content, which is common in
the UAE. On the image side, systems can call a true post "fake" just because the photo
*looks* AI-generated (FALSDO, IJCAI 2026).

**What we build.** **TwinCheck**, a bilingual (Arabic–English) evidence-grounded
fact-checking assistant. Input: a message (and optionally an image). Output: for each
claim, **Supported / Contradicted / Unverified**, a calibrated confidence, and a short
note citing the exact evidence sentence.

**Our contribution: "twins".** Copies of a real claim that change **exactly one
thing**, so we can test whether the verdict changes *for the right reason*:

| Twin | Suffix | What changes | Correct verdict | Made by |
|---|---|---|---|---|
| Paraphrase | `_P` | claim reworded, same meaning | same as original | Rathi |
| Arabic | `_A` | claim (and evidence) translated to Arabic | same as original | Youssef |
| Mixed | `_M` | claim in Arabic, evidence stays English | same as original | Youssef |
| Evidence swap | `_B` | one entity (place / date / person) swapped in every snippet that mentions it | **contradicted** | Rathi |
| Unrelated | `_C` | evidence replaced by snippets from an unrelated claim | **unverified** | Rathi |
| Image twin (MAAI901) | `_V` | photo passed through Stable Diffusion VAE (same content, AI fingerprints) | same as original | Feba |

We (1) **audit** existing approaches with the twins, then (2) **train** our model on
twins so it relies on evidence, and (3) test in **English, Arabic and mixed**.

**Related work we position against:** VitaminC (Schuster et al., NAACL 2021) did
evidence-change tests only on clean English Wikipedia; CCN (Abdelnabi et al., CVPR
2022) used real news + web evidence but only compared with vs without evidence.
FakeWorld (ICML 2026) already unifies authenticity + truth, so that is not our claim.
Always say "to our knowledge" when claiming novelty.

**Two courses, one system.**
- **MAAI902 (NLP & LLMs):** text pipeline, evidence and language twins, Arabic, explanations. Report lead: Himasri.
- **MAAI901 (Deep Learning for Visual Data):** image side, image twins, AI-image detector, CNN vs ViT, Grad-CAM, ONNX. Report lead: Feba.

---

## 3. Architecture

**Text pipeline (MAAI902)**

| Step | Job | Model | Trained? | Owner |
|---|---|---|---|---|
| 1 | Claim extraction from a messy message | Qwen3-8B-Instruct few-shot; LoRA Qwen3-4B for comparison | prompted / LoRA | Youssef |
| 2 | Evidence retrieval for the demo (offline library, no live web search) | BM25 + BGE-M3 (FAISS) over all cleaned snippets | frozen | Rathi |
| 3 | Entities (who / where / when) | spaCy `en_core_web_trf` | frozen | Rathi |
| 4 | Relevance of each snippet | `BAAI/bge-reranker-v2-m3` | fine-tuned 1 epoch | Rathi |
| 5 | NLI per snippet: entail / contradict / neutral | mDeBERTa-v3-base MNLI/XNLI checkpoint (e.g. `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`); XLM-R-large XNLI as comparison | fine-tuned | Rathi |
| 6 | Verdict head | MLP (in → 64 → 3, dropout 0.2) on ~10 features | trained with twin loss | Himasri |
| 7 | Calibration | temperature scaling + "Unverified" threshold | fitted | Himasri |
| 8 | Cited explanation | Qwen3-8B, cites only retrieved snippets [1]–[3] | prompted | Rathi |
| 9 | Faithfulness of the explanation | the fine-tuned NLI model | — | Rathi |

**Verdict-head features:** relevance-weighted support / contradict / neutral; max
support; max contradict; share of relevant snippets; entity match for place, date,
person (multilingual embedding similarity, so Arabic names match English ones).

**Verdict-head loss:** cross-entropy on originals + KL(original ‖ `_P`/`_A`/`_M`)
(keep the answer) + cross-entropy pushing `_B` → contradicted and `_C` → unverified.
λ grid: λ_inv ∈ {0, 0.5, 1, 2}, λ_sens ∈ {0, 0.5, 1}; choose by validation
consistency + ESR with ≤ 1 point macro-F1 loss. Adam, lr 1e-3, ≤ 30 epochs,
early stopping. Batches keep each original with its twins.

**Key design rule:** the verdict head **never sees** the "is it AI-made?" output, so
an AI-looking image cannot by itself turn a true post into "fake".

**Image side (MAAI901)**

| Step | Job | Model | Owner |
|---|---|---|---|
| V1 | Image + caption embeddings, image–caption match | SigLIP 2 (so400m) | Feba |
| V2 | Authenticity: camera / AI-generated / edited | logistic probe on CLIP ViT-L/14 (UnivFD-style) | Feba |
| V3 | CNN vs ViT detector comparison | ResNet-50 and ViT-B/16 (last 4 blocks unfrozen), augmentation | Feba |
| V4 | Image twins | SD VAE encode→decode (all true images) + SDXL img2img strength 0.15–0.25 (500 test) | Feba |
| V5 | Heatmaps | Grad-CAM | Feba |
| V6 | Deployment | ONNX export, ms per image | Feba |

---

## 4. Data (Plan B: public only)

| Dataset | Source | Size (approx.) | Use | Owner |
|---|---|---|---|---|
| MultiFC | Hugging Face mirror `pszemraj/multi_fc` or authors' release | ~35k claims, 26 fact-check sites, ≤10 Google snippets per claim | main English train/val/test, English twins, NLI + reranker pairs | Rathi |
| AVeriTeC | Hugging Face / FEVER workshop release | ~4.5k claims, QA-style web evidence | clean English test set | Rathi |
| X-FACT (Arabic part) | authors' GitHub release | ~31k claims in 25 languages | Arabic NLI training + Arabic test | Youssef |
| M4FC | GitHub `UKPLab/M4FC` | ~7k claims, ~5k images; ~860 Arabic claims | real Arabic test; images for vision | Youssef (images → Feba) |
| VERITE | GitHub `stevejpapad/image-text-verification` + image script | ~1,000 image–caption pairs | vision test set; image twins | Feba |
| GenImage (subset) | GenImage release | ~10k images per generator, 2–3 generators + real; hold one generator out | train the AI-image detector | Feba |

**MultiFC warnings.** The official test-split labels are **dummy values** (never
released): use train + validation only and make our own **80/10/10 split,
stratified by label, seed 42**. Labels differ per site (165 total). Remove the 202
duplicate claims. Licence listed as unknown: cite the paper, research use only.

**Check on download and post in the group:** are snippet texts included (not just
links)? Counts per label? Licence?

### Shared record format (JSON Lines, one record per line)

```json
{"id": "multifc_000123", "source": "multifc", "split": "train", "language": "en",
 "claim": "Dubai Metro suspended all lines today because of flooding",
 "evidence": [{"text": "Green Line operating normally, RTA confirms.", "url": "https://...", "domain": "gulfnews.com"}],
 "image_path": null,
 "label": "contradicted",
 "orig_label": "false"}
```

- `label` ∈ `supported`, `contradicted`, `unverified`. `language` ∈ `en`, `ar`, `mixed`.
- `id` starts with the source name and never changes; twins add the suffix (`_P`, `_A`, `_M`, `_B`, `_C`, `_V`).

### Label mapping (`data/label_map.csv`, owned by Rathi, Youssef adds Arabic labels)

- true, mostly true, correct, accurate → `supported`
- false, mostly false, pants on fire, fake, incorrect → `contradicted`
- unproven, unverified, not enough info → `unverified`
- half true, mixture, misleading, satire, unclear → **dropped** (report the count)

### Cleaning (`src/data/clean.py`, works on any dataset in the shared format)

1. Drop snippets from fact-checking sites: snopes.com, politifact.com, factcheck.org, fullfact.org, afp.com/factcheck, misbar.com, fatabyyano.net and similar.
2. Drop snippets with > 90% overlap with the claim.
3. Remove duplicate claims; keep ≤ 10 snippets per claim.
4. Keep claims with no snippets left in a separate file; report the count.
5. Use the dataset's split if it has real labels; otherwise 80/10/10, seed 42.

**Train data never touches test data. Twins keep the split of their source.**

### Twin rules

- Sources: `supported` claims with ≥ 2 snippets of relevance > 0.3; **500 test + 3,000 train**; list fixed with seed 42 in `data/copies/en/sources.csv`.
- `_B`: swap place→place, person→person, date shifted 2–5 years, using values from our own data. Nothing else changes.
- `_C`: evidence from a claim with no shared entities and reranker score < 0.1.
- `_P`: Qwen3-8B "same meaning, same names, dates and numbers"; keep only if spaCy entities match.
- Arabic twins: swaps are done in English **before** translation; translate with Qwen3-8B + entity glossary; 200 of the 500 test sources.
- Himasri + Feba hand-check 150 test twins (75 each, 30 overlap) → Cohen's kappa ≥ 0.7.

---

## 5. Metrics (all in `src/eval/metrics.py`, everyone calls the same functions)

| Metric | Definition | Target |
|---|---|---|
| Macro-F1, AUROC | on original claims | higher |
| Evidence sensitivity (ESR) | % `_B` predicted contradicted | → 100% |
| Unrelated rate | % `_C` predicted unverified | → 100% |
| Paraphrase consistency | % `_P` same verdict as original | → 100% |
| Cross-lingual consistency | % `_A`/`_M` same verdict as English original | → 100% |
| Authenticity leakage (ALR) | % `_V` whose verdict changes | → 0% |
| ECE | calibration error, 15 bins | lower |
| Faithfulness | % explanation sentences entailed by cited snippet | → 100% |
| Citation accuracy | % citations pointing to a supporting snippet | → 100% |
| Claim-extraction F1 | predicted vs gold claims, match if BGE-M3 sim ≥ 0.8 | higher |
| Retrieval recall@5 | offline index finds the claim's own snippets in top 5 | higher |

Also: paired bootstrap (1,000 resamples) 95% CIs + p-values on main comparisons;
seeds 42/43/44, report mean ± sd; human evaluation of 60 explanations (T1).

Predictions file format for every model: `results/<task-id>/preds_<model>.jsonl`
with `{"id", "pred_label", "probs": {"supported", "contradicted", "unverified"}}`.

**Baselines (Feba):** claim-only (no evidence), zero-shot NLI, similarity +
LightGBM (MUSE-style), LLM judge (Qwen3-8B prompted). Our model also vs an
ablation without reranker, without fine-tuning, without twin loss, without entity match.

---

## 6. Repo, folders and GitHub workflow

```
twincheck/
  data/raw/  data/processed/{en,ar,vision}/  data/copies/{en,ar,vision}/
  data/nli/  data/claims/  data/label_map.csv
  features/  models/  results/<task-id>/  reports/{tables,figures}/
  src/{data,copies,retrieval,nli,verdict,explain,claims,vision,translate,eval,demo}/
  src/pipeline.py   docs/implementation.md   notes/   references.bib
  requirements.txt  run_all.sh  README.md  .gitignore
```

- Python 3.11, `requirements.txt` (torch, transformers, sentence-transformers, datasets, spacy, rank_bm25, faiss-cpu, scikit-learn, lightgbm, wandb, gradio, diffusers, open_clip_torch, onnxruntime, peft).
- **Data, features and models are shared through git** (no shared Drive). Push them under `data/`, `features/`, `models/` with the paths in this file. GitHub refuses any single file over 100 MB (warns over 50 MB): split large outputs (e.g. one file per split) or track them with Git LFS (`git lfs track "*.safetensors"`). Raw downloads (`data/raw/`) and staging files stay out of git; rebuild them with the download scripts. The repo is public: only push data whose licence allows it (MultiFC: research use, cite the paper).
- **Branch per task:** `<name>/<task-id>-<short-name>`, e.g. `rathi/R3-twins`.
- **Commits start with the task ID:** `[R3] Add evidence-swap twin generator (Copy B)`.
- **Results:** small files only in `results/<task-id>/` (CSV/JSON, PNG, < 10 MB) + `README.md`.
- **Notebooks:** finished code moves into `src/*.py`; pushed notebooks have outputs cleared.
- **Pull request** to `main` titled `[R3] English twins`: what was built, how to run, output paths, key numbers. The **author reviews and merges it themselves** after running the tests and the 10-row sample (no second reviewer needed); post the merged PR link in the group.
- **By 23:00 every day:** push your branch, update Status in the tracker, add a line to `notes/contributions.md`.
- **Post a 10-row sample** of every hand-off file before the full file.

---

## 7. People and GPUs

There are **no sub-teams and no shared Colab accounts**. Everyone works individually,
on their own schedule, in their own Colab (or Kaggle) account. Collaboration happens
only through GitHub (pushed files, PRs) and the group chat.

| Person | Owns | GPU |
|---|---|---|
| Rathi (code) | data, NLP twins, NLI, reranker, explanations, demo | own Colab account |
| Youssef (all Arabic + claim extraction) | Arabic data, Arabic twins, claim extraction, Arabic experiments | own Colab account |
| Himasri (integration, eval, MAAI902 lead) | metrics, verdict head, pipeline, MAAI902 report | own Colab account (+ Kaggle) |
| Feba (vision, baselines, MAAI901 lead) | baselines, image side, MAAI901 report | own Colab account (+ Kaggle) |

Wherever this file says "Account A", "Account B" or that a job runs for someone else,
read it as: **each person runs their own jobs in their own account.**

**Rules:** post in the group when a hand-off file is pushed (path + commit); overnight
jobs need the browser tab open and the laptop awake; save checkpoints every ~20 minutes (to the Colab
session, and download/push the last one so a disconnect does not lose the run); stop the
runtime when the job ends; run Qwen3-8B in 4-bit or with vLLM; base models download
from Hugging Face into `/content`, only our fine-tuned checkpoints go to `models/` in git
(Git LFS if a file is over 100 MB).

**First cell of every Colab notebook**

```python
!git clone https://<github-token>@github.com/<org>/twincheck.git /content/twincheck
%cd /content/twincheck
!git checkout <name>/<task-id>-<short-name>
!pip install -q -r requirements.txt
from google.colab import userdata
import os, wandb
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')      # keys live in Colab Secrets
wandb.login(key=userdata.get('WANDB_KEY'))
```

---

## 8. Hand-offs (the only points where people wait on each other)

| From | File | To | Due |
|---|---|---|---|
| Rathi R1 | `data/processed/en/*.jsonl`, `data/label_map.csv`, `src/data/clean.py` | Himasri, Youssef, Feba | Mon 01:30 (sample), full by Tue 09:00 |
| Youssef Y1 | `data/processed/vision/m4fc_images/` | Feba F2 | Tue 01:00 |
| Rathi R2 | `features/relevance_en.parquet`, `features/entities_en.jsonl` | Rathi R3, Himasri H3 | Tue 09:30 |
| Himasri H2 | `data/nli/gold_val.jsonl` (150 EN) | Rathi R4 | Tue 12:00 |
| Youssef Y2 | `data/nli/ar_train.jsonl`, gold AR rows | Rathi R4 | Tue 14:00 |
| Rathi R3 | `data/copies/en/{test,train}.jsonl`, `sources.csv` | Youssef Y4, Himasri, Feba | Tue 14:00 |
| Youssef Y3 | `src/claims/extract.py` | Himasri H4, Rathi R6 | Tue 18:00 |
| Rathi R4 | `models/nli_v1/`, `features/nli_*.parquet` | Himasri, Feba, Youssef | Wed 09:00 |
| Himasri H4 | `src/pipeline.py` → `check(message, evidence=None, lang=None)` | Rathi R6, Youssef Y5 | Wed 13:00 |
| Feba F4 | `src/eval/baselines/*.py` | Youssef Y5 | Wed 13:00 |
| Rathi R8 | `results/T1/sheet.csv`; `reports/tables/`, `reports/figures/` | everyone; writers | Thu 10:00; Thu 20:00 |

If a hand-off is late, build against the 10-row sample and rerun later. Say so in the
group before the due time, with a new ETA.

**Milestones:** results freeze **Wed 18, 23:00** · everything finished **Sun 22, 21:00**
in `Final/` in the repo · rehearsals **Mon 23** · submit **Tue 24 by noon**.
Calls: optional; schedules differ, so status goes in the group chat and the tracker instead. Hand-offs happen by pushing to GitHub and posting the path.

---

## 9. Every person's tasks (steps)

### Rathi (code)

- **R1 · Mon 23:00–01:30 · English data.** Download MultiFC + AVeriTeC to `data/raw/`; `src/data/load_multifc.py`, `load_averitec.py` (each AVeriTeC QA answer → one snippet) → shared format; write `data/label_map.csv`; `src/data/clean.py` (rules in §4); own 80/10/10 split for MultiFC; save `data/processed/en/{multifc,averitec}_{train,val,test}.jsonl`; post 10-row sample + counts; `notes/R1.md` with counts per label/split and how many snippets each cleaning rule removed.
- **R2 · Mon 01:30 → overnight · Reranker + entities.** Fine-tune `BAAI/bge-reranker-v2-m3` 1 epoch on ~20k pairs (own snippets = relevant, random claim's snippets = irrelevant; lr 2e-5) → `models/reranker_v1/`; score every (claim, snippet) → `features/relevance_en.parquet` (`id, snippet_idx, relevance`); spaCy entities (PERSON, GPE, LOC, ORG, DATE, EVENT) → `features/entities_en.jsonl`; reusable `src/retrieval/rerank.py`, `src/data/entities.py`.
- **R3 · Tue 09:30–14:00 · English twins.** Sources list (§4) → `sources.csv`; `_P` paraphrase, `_B` swap, `_C` unrelated (rules in §4) for 500 test + 3,000 train; auto-checks; save `data/copies/en/{test,train}.jsonl`; post 10 examples of each type.
- **R4 · Tue 14:00 → Wed 09:00 · NLI.** ~30k balanced English weak pairs (supported claim + relevant snippet → entail; contradicted claim + relevant snippet → contradict; random snippet → neutral; `_B` snippets → contradict) + ~3k Arabic pairs; fine-tune mDeBERTa-v3 (lr 2e-5, batch 16, max len 256, 2 epochs, warmup 10%, fp16, seed 42); macro-F1 on 300 gold pairs (EN and AR separately) zero-shot vs fine-tuned; score every pair in all splits and twins → `features/nli_*.parquet` (`id, snippet_idx, p_entail, p_contradict, p_neutral`); XLM-R-large comparison overnight (own account).
- **R5 · Wed 09:30–16:00 · Explanations + offline index.** `src/explain/generate.py` (claim, verdict, top-3 snippets [1]–[3]; 1–2 sentences in the claim's language, cite only those numbers); `src/explain/faithfulness.py` (NLI on cited snippet vs each sentence; faithfulness + citation accuracy on 200 test claims); `src/retrieval/index.py` (BM25 + BGE-M3 FAISS over all cleaned snippets from MultiFC, AVeriTeC, X-FACT AR; top 20 → reranker → best 5; recall@5 on 200 claims); save 5 good and 5 bad notes.
- **R6 · Wed 16:00–19:00 · Demo.** `src/demo/app.py` (Gradio): message → claims → offline evidence → verdict per claim + cited note + confidence; 6 ready examples (2 Supported, 2 Contradicted, 2 Unverified; ≥ 2 Arabic, 1 mixed); made-up UAE examples labelled "illustrative example" on screen.
- **R7 · Wed 19:00–23:00 · Cleanup.** `run_all.sh`, README draft, fix teammates' bugs before freeze.
- **R8 · Thu · Figures + reproducibility.** By 10:00 sample 60 explanations (40 EN, 20 AR, seed 42) → `results/T1/sheet.csv`; `src/eval/figures.py` → every table (LaTeX + CSV) and figure (PNG 300 dpi) in `reports/`; fresh-clone test of `run_all.sh`.
- **R9 · Fri · Technical appendix.** `docs/implementation.md` (every model, checkpoint, hyperparameters, training time, GPU and units used, data sizes, twin counts); docstrings, folder READMEs, unit tests (10 rows each); tag `v0.9`.
- **R10 · Sat · Demo polish + video + slides.** Arabic right-to-left, confidence bar, cited links; 2–3 min demo video; 3 slides (data + twins, demo, repo); technical review of the MAAI901 draft.
- **R11 · Sun · Repo v1.0.** Merge all branches; `run_all.sh` reproduces main tables; README with results table; tag `v1.0`; check contribution log.

### Youssef (all Arabic + claim extraction)

- **Y1 · Mon 23:00 → Tue 10:30 · Arabic data.** X-FACT (Arabic) + M4FC → shared format, `language: ar`; M4FC images to `data/processed/vision/m4fc_images/` for Feba by 01:00; add Arabic-site labels to `label_map.csv` (sync with Rathi); run `clean.py` with Arabic fact-check sites added; save `data/processed/ar/{xfact_ar,m4fc_ar}_*.jsonl` (M4FC = test only); post counts.
- **Y3 (part 1) · Mon 01:00–02:00 + Tue 14:00–15:30 · Claim-extraction examples.** 300 WhatsApp-style messages (~100 EN, ~100 AR, ~100 mixed), each combining 1–3 real claims, with gold claims → `data/claims/gold.jsonl` (`id, language, message, claims[]`); LLM can draft, you check every one.
- **Y2 · Tue 10:30–14:00 · Arabic NLI data.** Weak pairs from X-FACT Arabic train (top up to ~3k by translating English pairs with Qwen3-8B + glossary) → `data/nli/ar_train.jsonl`; hand-label 150 Arabic gold pairs → append to `data/nli/gold_val.jsonl` with `language: ar`. Due 14:00.
- **Y3 (part 2) · Tue 15:30–18:00 · Extractor.** `src/claims/extract.py`: Qwen3-8B few-shot (6 examples, 2 per language), JSON list of claims in the original language; F1 on 200 held-out examples per language. **Wed 10:00–12:00:** LoRA Qwen3-4B on the other 100, compare.
- **Y4 · Tue 18:30 → Wed 10:00 · Arabic twins (own account).** 200 of the 500 test sources: `_A` (claim + snippets translated), `_M` (Arabic claim, English snippets), and Arabic versions of their `_B`/`_C`; check 50 translations; `data/copies/ar/test.jsonl`; error notes in `notes/Y4.md`.
- **Y5 · Wed 12:00–23:00 · Arabic experiments E7.** Our pipeline + Feba's baselines on Arabic twins, mixed twins, X-FACT AR test, M4FC AR; per language (EN / AR / mixed): macro-F1, ESR, unrelated rate, cross-lingual consistency → `results/Y5/arabic_E7.csv`.
- **Y6 · Wed eve + Thu · Arabic error analysis + sections.** 20 failures by cause (dialect, translation, name spelling, missing evidence, other); T1 rate 20 Arabic notes; Arabic section for each report, worded differently, to Himasri + Feba by Thu 20:00.
- **Y7 · Fri · Arabic integration.** Place Arabic sections in both drafts (10:00 Himasri, 11:00 Feba); Arabic slide images; Arabic part of presentation script; comment on Arabic terms.
- **Y8 · Sat · Numbers check.** Check every number in both report drafts against `results/` (list to writers by 14:00); Arabic human-eval paragraph; Arabic slides.
- **Y9 · Sun · Sign-off.** Demo with 5 Arabic messages; papers vs reports consistency; final Arabic check of final docs + slides; post sign-off.

### Himasri (integration, evaluation, MAAI902 lead)

- **H1 · Mon 23:00–02:00 · Repo + metrics.** Repo with the §6 layout, `requirements.txt`, `.gitignore`, W&B project, Overleaf projects (one per course), `references.bib`, `notes/contributions.md`; invite all; `src/eval/metrics.py` (all §5 metrics) + unit tests on fake predictions; `src/eval/report.py` (predictions → CSV + LaTeX tables).
- **H2 · Tue 09:10–13:00 · Gold pairs + skeleton.** Hand-label 150 English (snippet, claim) pairs from the R1 val split → `data/nli/gold_val.jsonl` (`language: en`) by 12:00; MAAI902 report skeleton (8 sections).
- **H3 · Tue 13:15–20:30 · Verdict head.** Zero-shot NLI scores (base mDeBERTa); `src/verdict/features.py`, `model.py`, `train.py` with twin loss (§3); train v0 without twin terms; run `metrics.py`.
- **H4 · Wed 08:00–13:00 · Pipeline + calibration.** 08:00 hand-check 75 twins (30 overlap with Feba) → kappa; `src/pipeline.py` `check(message, evidence=None, lang=None)` using Youssef's extractor and fine-tuned NLI; λ grid; temperature scaling + Unverified threshold; settings in `notes/H4.md`.
- **H5 · Wed 13:15–23:00 · Experiments.** E1 our model vs baselines on MultiFC test + AVeriTeC; E2 twin metrics with vs without twin loss; E5 calibration on MultiFC, AVeriTeC, X-FACT AR; ablations (no reranker, zero-shot NLI, no twin loss, no entity match); seeds 42/43/44; bootstrap CIs + p-values; freeze 23:00.
- **H6 · Thu · Writing + analysis.** English error analysis (10 cases); T1 rate 40 English notes; compute kappa + averages (16:00); MAAI902 introduction, related work, methods, results.
- **H7 · Fri · MAAI902 report full draft** (8 sections) by 21:00.
- **H10 · Sat · MAAI902 paper** in Overleaf (Literature Review + Ablation Studies sections, contribution statement), written fresh; review Feba's MAAI901 draft 09:00–10:30.
- **H8 · Sun · Finalise.** Report + paper from comments and Youssef's mismatch list; Turnitin < 10%; assemble the full slide deck (slides from everyone by 15:00); upload to `Final/` by 21:00.
- **H9 · Tue 24 · Submit** all four documents by noon after a final read with Feba.

### Feba (vision, baselines, MAAI901 lead)

- **F1 · Mon 23:00–02:00 · Vision data.** VERITE (+ image script; count failures); check if RED-DOT VERITE evidence is public; GenImage subset (real + 2–3 generators, ~10k each, one generator held out); VERITE → `data/processed/vision/verite.jsonl`.
- **F2 · Mon night → Tue 11:30 · Features + image twins.** Cache SigLIP 2 + CLIP ViT-L/14 embeddings (VERITE, M4FC, GenImage) → `features/vision/*.npy`; `_V` twins: VAE round-trip for all true images + SDXL img2img (0.15–0.25) for 500 test; save originals and twins as JPEG 85; keep twins with SigLIP cosine ≥ 0.92; check 30 by eye; skip large identifiable faces.
- **F3 · Tue 11:30–18:30 · Detector + leakage + skeleton.** MAAI901 report skeleton (12 sections); logistic probe on CLIP features (camera / AI / edited), seen vs unseen generator; leakage test on VERITE: fused model (sees image embedding + authenticity) vs our decoupled model → ALR; claim-only + similarity baselines on English originals.
- **F4 · Wed 08:00–13:00 · Baselines audit.** 08:00 hand-check 75 twins; baselines in `src/eval/baselines/` (claim-only, zero-shot NLI, similarity + LightGBM, LLM judge on Kaggle) on originals + all English twins → `results/F4/audit_E0.csv`; scripts take any twin file so Youssef can rerun them on Arabic.
- **F5 · Wed 13:15–23:00 · Vision experiments.** Grad-CAM (8 examples); robustness (JPEG 50/70, 20% crop); ONNX export + ms per image; **CNN vs ViT:** ResNet-50 and ViT-B/16 (last 4 blocks) on GenImage with augmentation (crop, flip, JPEG 60–95, blur), 3 epochs each, vs CLIP probe on seen/unseen generators and under JPEG/crop; freeze 23:00.
- **F6 · Thu · Writing + analysis.** Vision error analysis (8 cases with Grad-CAM); T1 rate 40 English notes; MAAI901 introduction, related work, methods, results.
- **F7 · Fri · MAAI901 report full draft** (12 sections; augmentation + image-twin pipeline in Data section; CNN vs ViT table) by 21:00.
- **F9 · Sat · MAAI901 paper** in Overleaf (visual results, CNN vs ViT, leakage, contributions), written fresh; review Himasri's MAAI902 draft 09:00–10:30.
- **F8 · Sun · Finalise.** Report + paper; Turnitin < 10% and no paragraphs shared with MAAI902; vision slides to Himasri by 15:00; upload to `Final/` by 21:00.

### Everyone

- **T1 · Thu · Human evaluation.** 60 notes (40 EN rated by Himasri + Feba, 20 AR by Youssef): verdict correct (yes/no), every sentence supported by its cited source (yes/partly/no), readability (1–5) → `results/T1/human_eval.csv`.
- **P1 · Mon 23 · Preparation.** Read final documents; rehearsals 13:00 and 21:00 (timed); small fixes only.

---

## 10. Writing rules

- MAAI902 report + paper: Himasri. MAAI901 report + paper: Feba. Arabic sections: Youssef. Code details for methods: Rathi's `docs/implementation.md` and everyone's `notes/<task-id>.md`.
- **Never copy paragraphs between the two courses' documents or between a report and its paper** (Turnitin < 10% each).
- Each document has a team contribution statement (from `notes/contributions.md`).
- Every number in a document must come from a file in `results/`.

---

## 11. Ground rules

1. Shared format and file names exactly; never edit another person's files or raw data.
2. Train data never touches test data.
3. Sample first (10 rows), then the full file.
4. If it is not logged in W&B, it did not happen.
5. Push daily to your own branch; data, features and models go in git too (no file over 100 MB without Git LFS; no `data/raw/`).
6. Blocked for 2 hours? Ask in the group (name the person whose file you need).
7. Changes to datasets, models or this plan are agreed in the group first.
8. Results freeze Wed 18, 23:00. After that: analysis, writing and fixes only.
