# TwinCheck

A bilingual (Arabic–English) fact-checking assistant grounded in evidence. For each
claim in a message it returns **Supported / Contradicted / Unverified**, a calibrated
confidence, and a short note citing the evidence sentence it relies on. We test whether
models read the evidence by using **twins**: copies of a claim that change exactly one
thing (paraphrase, Arabic, mixed language, swapped entity in the evidence, unrelated
evidence, AI-fingerprinted image).

UOWD group project for MAAI902 (NLP & LLMs) and MAAI901 (Deep Learning for Visual Data).

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

Data, features and models are shared through git under `data/`, `features/` and
`models/` (see `data/README.md`). Raw downloads are not: rebuild them with the
download scripts. Files over 100 MB need Git LFS.

## Folder map

| Folder | Contents |
|---|---|
| `src/data/` | loaders + `clean.py` (shared record format) |
| `src/copies/` | twin generators (`_P _B _C`, Arabic `_A _M`) |
| `src/retrieval/` | reranker, offline BM25 + FAISS index |
| `src/nli/` | NLI fine-tuning and scoring |
| `src/verdict/` | verdict-head features, model, twin-loss training |
| `src/explain/` | cited explanations + faithfulness |
| `src/claims/` | claim extraction |
| `src/vision/` | image embeddings, authenticity detector, image twins |
| `src/eval/` | `loaders.py` (format checks), `metrics.py` (all metrics), `report.py` (tables), `baselines/` |
| `src/demo/` | Gradio demo |
| `src/pipeline.py` | `check(message, evidence=None, lang=None)`: the end-to-end entry point |
| `results/<task-id>/` | small result files + `README.md` |
| `reports/{tables,figures}/` | generated LaTeX/CSV tables and figures |
| `notes/` | per-task notes and `contributions.md` |
| `tests/` | unit tests (10-row fixtures) |

## Evaluation

Every model writes `results/<task-id>/preds_<model>.jsonl`, one line per record:
`{"id", "pred_label", "probs": {"supported", "contradicted", "unverified"}}`.

```bash
python -m src.eval.report --gold data/processed/en/multifc_test.jsonl data/copies/en/test.jsonl
```

## How the text pipeline works

`check(message)` in `src/pipeline.py` runs these steps for every claim in a message:

| Step | What happens | Code | Model |
|---|---|---|---|
| 1 | split the message into claims | `src/claims/extract.py` | Qwen3-8B few-shot |
| 2 | find evidence in the offline library (BM25 + dense top 20 → reranker → best 5) | `src/retrieval/index.py` | BGE-M3 + fine-tuned `bge-reranker-v2-m3` |
| 3 | NLI for each snippet: entail / contradict / neutral | `src/nli/score.py` | mDeBERTa-v3 XNLI, zero-shot (our fine-tuned v2 tied with it, see R4) |
| 4 | verdict from about 10 features, trained with the twin loss | `src/verdict/infer.py` | small MLP (`models/verdict_twin_s42`) |
| 5 | temperature scaling + "Unverified" threshold | `src/verdict/calibrate.py` | — |
| 6 | 1–2 sentence note citing snippets [1]–[3] | `src/explain/generate.py` | Qwen3-8B |

Each step falls back to a simple stand-in when its module or model is missing, so
`python -m src.pipeline --status` shows which parts are real.

## Models and large files

These are too big for the repo's Git LFS quota, so they are on the Hugging Face Hub.
The code downloads them by itself:

| What | Where | Made in |
|---|---|---|
| fine-tuned reranker | `RathiAbeth/twincheck-reranker-v1` | R2 |
| fine-tuned NLI model (v2; v1 kept for the record) | `RathiAbeth/twincheck-nli-v2`, `RathiAbeth/twincheck-nli-v1` | R4 |
| offline evidence library (snippets + BGE-M3 embeddings) | dataset `RathiAbeth/twincheck-index-v1`; `python -m src.retrieval.build_index download` | R5 |

## Reproduce

```bash
bash run_all.sh                        # unit tests + every table and figure from results/ (CPU)
STAGES="nli tables" bash run_all.sh    # chosen stages
STAGES=all bash run_all.sh             # everything from the raw downloads (GPU, several hours)
```

Stages: `tests data features twins arabic nli verdict explain demo baselines vision tables`.
`vision` needs `GENIMAGE_ROOT` (the F1 GenImage images are not in git). Each GPU task also
has a Colab notebook in `notebooks/` that runs it and pushes the outputs. Training runs are
logged to W&B (project `twincheck`, seed 42) unless the task README says otherwise.
Settings, checkpoints, training times and GPUs for every model: `docs/implementation.md`.

## Demo

```bash
python -m src.retrieval.build_index download   # offline evidence library (~330 MB)
python -m src.demo.app                         # http://127.0.0.1:7860 (add --share in Colab)
```

The six ready examples are made-up UAE messages and are marked "Illustrative example"
in the app. Arabic is shown right-to-left, the citations [1]–[3] in each note link to their
sources, and the footer says which verdict step is running. With the trained verdict head
2 of 6 come out as expected; with the simple rule (`TWINCHECK_VERDICT=none`, max entail vs
max contradict) 5 of 6 do (`results/R6/README.md`). Colab: `notebooks/R10_demo_colab.ipynb`.

## Results

All tables and figures are in `reports/` and are rebuilt by `python -m src.eval.figures`.
Numbers below are copied from those files (results frozen Wed 18 Nov, 23:00).

**Our verdict head.** MultiFC test (2,106 claims) and its English twins, zero-shot NLI
features, mean ± sd over seeds 42/43/44, in % (`reports/tables/tab_main.csv`).

| Model | Macro-F1 | ESR | Unrelated | Para. cons. | ECE (calibrated) |
|---|---|---|---|---|---|
| verdict head, no twin loss | 36.3 ± 0.5 | 52.2 ± 3.0 | 42.4 ± 24.4 | 91.2 ± 1.7 | 5.1 ± 0.4 |
| **verdict head + twin loss** | 35.2 ± 0.7 | **89.7 ± 0.9** | **100.0 ± 0.0** | 91.4 ± 2.5 | 7.0 ± 0.3 |
| + twin loss, no reranker | 30.1 ± 1.4 | 68.7 ± 0.6 | 81.0 ± 3.8 | 87.4 ± 0.6 | — |
| + twin loss, no entity match | 33.8 ± 1.3 | 86.8 ± 3.9 | 100.0 ± 0.0 | 90.6 ± 0.9 | — |

- The twin loss raises ESR by 35–42 points and the unrelated rate by 32–81 points
  (paired bootstrap, p < 0.001 in every seed) for at most 2.3 points of macro-F1. Part of
  the ESR gain is a shift toward "contradicted": the twin model also calls 61% of the
  supported source claims contradicted (`results/H4/esr_vs_source_bias.json`).
- AVeriTeC test: macro-F1 38.3 ± 2.0 (twin) vs 38.0 ± 2.9 (no twin loss), no significant difference.
- Other components: claim extraction F1 98.4 (Y3); retrieval recall@5 EN 94.5 / AR 92.0 (R5);
  explanation faithfulness 32 (R5); NLI gold macro-F1 zero-shot 37.1 EN / 44.3 AR (R4).

**Baselines audit (E0).** Same MultiFC test and twins, in % (`reports/tables/tab_audit_E0.csv`, F4).

| Baseline | Macro-F1 | ESR | Unrelated | Para. cons. |
|---|---|---|---|---|
| claim-only (no evidence) | 48.1 | 44.5 | 1.2 | 84.5 |
| similarity + LightGBM (3 seeds) | 38.5 ± 0.5 | 59.2 ± 1.1 | 1.4 ± 0.2 | 73.0 ± 2.5 |
| zero-shot NLI | 34.6 | 62.3 | 53.6 | 87.4 |
| LLM judge (Qwen3-8B) | 23.3 | 10.0 | 95.0 | 94.8 |

- The claim-only model has the best macro-F1 while never seeing the evidence, and it almost
  never says "unverified" when the evidence is swapped for unrelated snippets (1.2%). This
  is the shortcut the twins are built to expose.

**Vision (MAAI901).** AI-image detectors trained on real + BigGAN, tested on real + unseen
Stable Diffusion v1.4, seed 42 (`reports/tables/tab_detectors.csv`, F3/F5).

| Detector | Accuracy | Recall (AI) | AUROC |
|---|---|---|---|
| CLIP ViT-L/14 + logistic regression (frozen) | 0.510 | 0.024 | 0.860 |
| CLIP ViT-L/14 + linear SVM (frozen) | 0.507 | 0.014 | 0.903 |
| ResNet-50 (last 4 blocks) | 0.505 | 0.012 | 0.713 |
| ViT-B/16 (last 4 blocks) | 0.538 | 0.080 | 0.885 |

- All detectors rank unseen-generator images well (AUROC up to 0.90) but flag almost none at
  the threshold learned on BigGAN. On VERITE, adding the authenticity score to the verdict
  model changes no pair verdicts (ALR 7.8% on 51 test pairs for both, `tab_leakage.csv`).
  ONNX: 15.5 ms (ResNet-50) / 85.7 ms (ViT-B/16) per image on CPU (`tab_onnx.csv`).
- Arabic experiments (Y5, `tab_arabic_E7`) are added when `results/Y5/arabic_E7.csv` is pushed.

## Team

Rathi (code: data, twins, reranker, NLI, explanations, demo) · Youssef (Arabic +
claim extraction) · Himasri (integration + evaluation, MAAI902 lead) · Feba (vision +
baselines, MAAI901 lead). Contributions are logged in `notes/contributions.md`.
