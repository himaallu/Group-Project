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

## Results

_To be filled after results freeze (Wed 18 Nov, 23:00)._

| Model | Macro-F1 | ESR | Unrelated | Para. cons. | X-ling cons. | ECE |
|---|---|---|---|---|---|---|
| | | | | | | |

## Team

Rathi (Team B, code) · Youssef (Team B, Arabic + claim extraction) ·
Himasri (Team A, integration + evaluation, MAAI902 lead) · Feba (Team A, vision +
baselines, MAAI901 lead). Contributions are logged in `notes/contributions.md`.
