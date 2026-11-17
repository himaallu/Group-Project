#!/usr/bin/env bash
# TwinCheck: reproduce the project, stage by stage. Owner: Rathi (R7); stub from H1.
#
# What it does
#   Runs the stages listed in $STAGES, in this order:
#     tests     unit tests (10-row fixtures, CPU, ~10 s)
#     data      R1 English data from the public downloads (CPU, ~5 min)
#     features  R2 reranker fine-tuning, relevance scores, spaCy entities (GPU)
#     twins     R3 English twins _P _B _C + checks + twin features (GPU, ~1 h)
#     nli       R4 weak NLI pairs, mDeBERTa fine-tuning, NLI scores; H3 zero-shot scores (GPU)
#     verdict   H3/H4 verdict head (with and without twin loss, seeds 42-44) + calibration
#     explain   R5 cited notes, faithfulness, offline index + recall@5 (GPU)
#     tables    every table and figure in reports/ from the files in results/ (CPU)
#   A step whose script is not in the repo yet is skipped with a message, so the
#   script works while teammates' parts are still arriving.
#   Arabic data (Y1) needs manual raw downloads; see notes/Y1.md. Its outputs are in git.
#
# Inputs
#   The repo (data/, features/ and results/ are in git); internet for model downloads.
#   GPU stages expect a CUDA GPU (Colab A100/L4/T4) and HF_TOKEN / W&B login for uploads.
#
# Outputs
#   The files named in each task's results/<task-id>/README.md; tables and figures in
#   reports/tables/ and reports/figures/.
#
# How to run (repo root)
#   bash run_all.sh                                   # default: tests + tables (CPU, minutes)
#   STAGES="tests nli tables" bash run_all.sh         # chosen stages
#   STAGES=all bash run_all.sh                        # everything (GPU, several hours)
set -euo pipefail

STAGES="${STAGES:-tests tables}"
if [[ "$STAGES" == "all" ]]; then
  STAGES="tests data features twins nli verdict explain tables"
fi
SEEDS="${SEEDS:-42 43 44}"
EN=data/processed/en
HUB_USER="${HUB_USER:-RathiAbeth}"   # owner of the Hugging Face copies of our models

# True if a Python module exists in the repo (e.g. src.eval.figures).
has() { python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" 2>/dev/null; }

# step "<label>" <module> [args...]: run python -m <module> args, or skip if missing.
step() {
  local label=$1 module=$2
  shift 2
  if has "$module"; then
    echo "[run_all] $label"
    python -m "$module" "$@"
  else
    echo "[run_all] skip $label ($module is not in the repo yet)"
  fi
}

wants() { [[ " $STAGES " == *" $1 "* ]]; }

if wants tests; then
  echo "[run_all] unit tests"
  python -m pytest -q
fi

if wants data; then
  echo "[run_all] R1 English data"
  bash src/data/run_r1.sh
fi

if wants features; then
  step "R2 reranker fine-tuning" src.retrieval.train_reranker
  step "R2 relevance scores" src.retrieval.rerank $EN/{multifc,averitec}_{train,val,test}.jsonl \
    --model models/reranker_v1 --out features/relevance_en.parquet
  step "R2 entities" src.data.entities $EN/{multifc,averitec}_{train,val,test}.jsonl \
    --out features/entities_en.jsonl
fi

if wants twins; then
  step "R3 twins" src.copies.make_twins --reranker models/reranker_v1
  step "R3 checks" src.copies.checks
  step "R3 twin relevance" src.retrieval.rerank data/copies/en/{test,train}.jsonl \
    --model models/reranker_v1 --out features/relevance_copies_en.parquet
  step "R3 twin entities" src.data.entities data/copies/en/{test,train}.jsonl \
    --out features/entities_copies_en.jsonl
fi

NLI_FILES="$EN/multifc_train.jsonl $EN/multifc_val.jsonl $EN/multifc_test.jsonl \
  $EN/averitec_train.jsonl $EN/averitec_val.jsonl $EN/averitec_test.jsonl \
  data/copies/en/test.jsonl data/copies/en/train.jsonl \
  data/processed/ar/xfact_ar_train.jsonl data/processed/ar/xfact_ar_val.jsonl \
  data/processed/ar/xfact_ar_test.jsonl data/processed/ar/m4fc_ar_test.jsonl"

if wants nli; then
  step "H3 zero-shot NLI scores" src.nli.score_zeroshot $NLI_FILES --out-dir features/nli_zeroshot
  step "R4 weak NLI pairs" src.nli.build_pairs
  step "R4 NLI fine-tuning" src.nli.train
  step "R4 fine-tuned NLI scores" src.nli.score $NLI_FILES --model models/nli_v1 \
    --out-dir features/nli_v1
fi

if wants verdict; then
  # Settings follow H3/H4; update the lambdas here after the H4 grid search.
  for seed in $SEEDS; do
    step "H3 verdict v0 (seed $seed)" src.verdict.train --name v0 --seed "$seed" \
      --nli features/nli_v1/multifc_{train,val,test}.parquet
    step "H3 verdict twin loss (seed $seed)" src.verdict.train --name inv1_sens1 --seed "$seed" \
      --lambda-inv 1 --lambda-sens 1 --twins data/copies/en/train.jsonl data/copies/en/test.jsonl \
      --nli features/nli_v1/multifc_{train,val,test}.parquet features/nli_v1/{train,test}.parquet
  done
  step "H4 calibration" src.verdict.calibrate --val-logits results/H3/val_logits_inv1_sens1_s42.jsonl \
    --out models/verdict_calib.json
fi

if wants explain; then
  export TWINCHECK_NLI="${TWINCHECK_NLI:-$HUB_USER/twincheck-nli-v1}"
  step "R5 cited notes" src.explain.generate --reranker "$HUB_USER/twincheck-reranker-v1"
  step "R5 faithfulness" src.explain.faithfulness
  step "R5 offline index" src.retrieval.build_index build
  step "R5 recall@5" src.retrieval.build_index eval --reranker "$HUB_USER/twincheck-reranker-v1"
fi

if wants tables; then
  if compgen -G "results/*/preds_*.jsonl" > /dev/null; then
    step "evaluation tables" src.eval.report \
      --gold $EN/multifc_test.jsonl data/copies/en/test.jsonl --out reports/tables
  else
    echo "[run_all] skip evaluation tables (no results/*/preds_*.jsonl yet)"
  fi
  step "R8 tables and figures" src.eval.figures
fi

echo "[run_all] done: $STAGES"
