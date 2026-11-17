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
| 3 | NLI for each snippet: entail / contradict / neutral | `src/nli/score.py` | fine-tuned mDeBERTa-v3 |
| 4 | verdict from about 10 features, trained with the twin loss | `src/verdict/` | small MLP |
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
| fine-tuned NLI model | `RathiAbeth/twincheck-nli-v1` | R4 |
| offline evidence library (snippets + BGE-M3 embeddings) | dataset `RathiAbeth/twincheck-index-v1`; `python -m src.retrieval.build_index download` | R5 |

## Reproduce

```bash
bash run_all.sh                        # unit tests + tables from the files in results/ (CPU)
STAGES="nli tables" bash run_all.sh    # chosen stages
STAGES=all bash run_all.sh             # everything from the raw downloads (GPU, several hours)
```

Each GPU task also has a Colab notebook in `notebooks/` (`R2_…` to `R6_…`) that runs it
and pushes the outputs. Every training run is logged to W&B (project `twincheck`, seed 42).

## Demo

```bash
python -m src.retrieval.build_index download   # offline evidence library (~330 MB)
python -m src.demo.app                         # http://127.0.0.1:7860 (add --share in Colab)
```

The six ready examples are made-up UAE messages and are marked "Illustrative example"
in the app.

## Results

_To be filled after results freeze (Wed 18 Nov, 23:00)._

| Model | Macro-F1 | ESR | Unrelated | Para. cons. | X-ling cons. | ECE |
|---|---|---|---|---|---|---|
| | | | | | | |

## Team

Rathi (code: data, twins, reranker, NLI, explanations, demo) · Youssef (Arabic +
claim extraction) · Himasri (integration + evaluation, MAAI902 lead) · Feba (vision +
baselines, MAAI901 lead). Contributions are logged in `notes/contributions.md`.
