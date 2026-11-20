#!/usr/bin/env bash
# TwinCheck: reproduce the project, stage by stage. Owner: Rathi (R7); stub from H1.
#
# What it does
#   Runs the stages listed in $STAGES, in this order:
#     tests     unit tests (10-row fixtures, CPU, ~10 s)
#     data      R1 English data from the public downloads (CPU, ~5 min)
#     features  R2 reranker fine-tuning, relevance scores, spaCy entities (GPU)
#     twins     R3 English twins _P _B _C + checks + twin features (GPU, ~1 h)
#     arabic    Y2 Arabic NLI pairs (translation on GPU), Y3 claim-extraction F1 (GPU)
#     nli       zero-shot NLI scores (default model); R4 v2 pairs, fine-tuning, scores (GPU)
#     verdict   H3 v0, H4 lambda grid + twin/notwin heads + calibration, H5 ablations, E5 Arabic
#     explain   R5 cited notes, faithfulness, offline index + recall@5 (GPU)
#     demo      R6 the 6 demo examples through the full pipeline (GPU)
#     baselines F4 claim-only, similarity + LightGBM, zero-shot NLI, LLM judge (GPU) + audit E0
#     vision    F5 ResNet-50 / ViT-B/16 detectors, robustness, Grad-CAM, ONNX (GPU; needs
#               GENIMAGE_ROOT = folder with the F1 GenImage images, which are not in git)
#     tables    every table and figure in reports/ from the files in results/ (CPU)
#   A step whose script is not in the repo yet is skipped with a message, so the
#   script works while teammates' parts are still arriving.
#   Arabic data (Y1) needs manual raw downloads; see notes/Y1.md. Its outputs are in git.
#   Vision data and features (F1-F3) run from Feba's notebooks (notebooks/Feba_F*.ipynb);
#   their outputs are in git.
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
  STAGES="tests data features twins arabic nli verdict explain demo baselines vision tables"
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

ZS=MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7   # zero-shot NLI (default model)
NLI_FILES="$EN/multifc_train.jsonl $EN/multifc_val.jsonl $EN/multifc_test.jsonl \
  $EN/averitec_train.jsonl $EN/averitec_val.jsonl $EN/averitec_test.jsonl \
  data/processed/ar/xfact_ar_train.jsonl data/processed/ar/xfact_ar_val.jsonl \
  data/processed/ar/xfact_ar_test.jsonl data/processed/ar/m4fc_ar_test.jsonl"
TWIN_FILES="data/copies/en/test.jsonl data/copies/en/train.jsonl"

if wants arabic; then
  # Y2 Arabic NLI pairs; the translation top-up needs a GPU (~45 min on a T4).
  step "Y2 Arabic weak pairs" src.nli.build_ar_pairs
  step "Y2 translated pairs" src.translate.translate_pairs --n 954
  step "Y2 merge pairs" src.translate.translate_pairs --merge
  # Y3 claim extraction F1 on the 200 held-out messages (rule baseline, then Qwen3-8B).
  step "Y3 claim extraction (rule)" src.claims.evaluate --backend split --no_wandb
  step "Y3 claim extraction (Qwen3-8B)" src.claims.evaluate --backend qwen --no_wandb
fi

if wants nli; then
  # Zero-shot NLI scores: the features every reported verdict-head result uses (H3/H4/H5).
  step "zero-shot NLI scores" src.nli.score $NLI_FILES --model "$ZS" --out-dir features/nli_zeroshot
  step "zero-shot NLI scores (twins)" src.nli.score $TWIN_FILES --model "$ZS" \
    --out-dir features/nli_zeroshot/copies_en
  # R4 v2: weak pairs filtered by the zero-shot model, fine-tuning, scores (ablation features).
  step "R4 NLI pairs (v2)" src.nli.build_pairs --filter-model "$ZS"
  step "R4 NLI fine-tuning (v2)" src.nli.train --version v2 --out models/nli_v2
  step "R4 fine-tuned NLI scores" src.nli.score $NLI_FILES $TWIN_FILES --model models/nli_v2 \
    --out-dir features/nli_v2
fi

if wants verdict; then
  ZNLI="features/nli_zeroshot/multifc_train.parquet features/nli_zeroshot/multifc_val.parquet \
    features/nli_zeroshot/multifc_test.parquet"
  for seed in $SEEDS; do
    step "H3 verdict v0 (seed $seed)" src.verdict.train --name v0 --seed "$seed" --no-wandb --nli $ZNLI
  done
  # H4: lambda grid (seed 42), then twin / notwin heads for every seed.
  step "H4 lambda grid" src.verdict.grid --no-wandb --twins $TWIN_FILES \
    --nli $ZNLI features/nli_zeroshot/copies_en/train.parquet features/nli_zeroshot/copies_en/test.parquet \
    --relevance features/relevance_en.parquet features/relevance_copies_en.parquet
  for seed in $SEEDS; do
    for name in twin notwin; do
      step "H4 calibration ($name, seed $seed)" src.verdict.calibrate \
        --val-logits "results/H4/val_logits_${name}_s${seed}.jsonl" \
        --out "models/verdict_calib_${name}_s${seed}.json" \
        --apply "results/H4/preds_${name}_s${seed}.jsonl" "results/H4/preds_${name}_cal_s${seed}.jsonl"
    done
  done
  [[ -f models/verdict_calib_twin_s42.json ]] && cp models/verdict_calib_twin_s42.json models/verdict_calib.json   # the pipeline's settings
  # H5: ablations + AVeriTeC + bootstrap; E5 on X-FACT Arabic.
  step "H5 experiments" src.eval.run_h5 --no-wandb
  step "H5 E5 Arabic" src.eval.run_e5_ar
fi

if wants explain; then
  export TWINCHECK_NLI="${TWINCHECK_NLI:-$ZS}"
  step "R5 cited notes" src.explain.generate --reranker "$HUB_USER/twincheck-reranker-v1"
  step "R5 faithfulness" src.explain.faithfulness
  step "R5 offline index" src.retrieval.build_index build
  step "R5 recall@5" src.retrieval.build_index eval --reranker "$HUB_USER/twincheck-reranker-v1"
fi

if wants demo; then
  # R6: the 6 ready examples through the full pipeline (GPU); `python -m src.demo.app --share` for the app.
  export TWINCHECK_NLI="${TWINCHECK_NLI:-$ZS}"
  step "R6 demo examples" src.demo.app --run-examples results/R6/examples.json
fi

if wants baselines; then
  # F4: baselines trained on MultiFC train, tuned on val, run on MultiFC test + English twins.
  BASE="--train $EN/multifc_train.jsonl --val $EN/multifc_val.jsonl"
  TEST="$EN/multifc_test.jsonl data/copies/en/test.jsonl"
  step "F4 claim-only" src.eval.baselines.claim_only $BASE --predict $TEST \
    --out results/F4/preds_claim_only.jsonl
  step "F4 similarity + LightGBM" src.eval.baselines.similarity $BASE --predict $TEST \
    --out-dir results/F4
  step "F4 zero-shot NLI" src.eval.baselines.zeroshot_nli --val-records $EN/multifc_val.jsonl \
    --val-nli features/nli_zeroshot/multifc_val.parquet --predict $TEST \
    --predict-nli features/nli_zeroshot/multifc_test.parquet features/nli_zeroshot/copies_en/test.parquet \
    --out results/F4/preds_zeroshot_nli.jsonl
  step "F4 LLM judge" src.eval.baselines.llm_judge --predict $TEST --out results/F4/preds_llm_judge.jsonl
  # the audit also reads the AVeriTeC predictions (preds_avt_*) already in results/F4
  step "F4 audit E0" src.eval.baselines.audit --preds results/F4/preds_*.jsonl \
    --gold $TEST $EN/averitec_test.jsonl --out results/F4/audit_E0.csv
fi

if wants vision; then
  if [[ -z "${GENIMAGE_ROOT:-}" ]]; then
    echo "[run_all] skip F5 vision (set GENIMAGE_ROOT to the F1 GenImage image folder)"
  else
    CKPT="${VISION_CKPT:-models/vision}"
    for arch in resnet50 vit_b16; do
      step "F5 train $arch" src.vision.train_detector --arch "$arch" --seed 42 \
        --root "$GENIMAGE_ROOT" --ckpt-dir "$CKPT" --resume --out results/F5 --no-wandb
    done
    step "F5 evaluate + robustness" src.vision.evaluate_detectors --root "$GENIMAGE_ROOT" \
      --ckpt-dir "$CKPT" --skip-clip
    step "F5 Grad-CAM" src.vision.gradcam --root "$GENIMAGE_ROOT" --ckpt-dir "$CKPT"
    step "F5 ONNX" src.vision.export_onnx --ckpt-dir "$CKPT"
  fi
fi

if wants tables; then
  if compgen -G "results/*/preds_*.jsonl" > /dev/null; then
    step "evaluation tables" src.eval.report \
      --gold $EN/multifc_test.jsonl data/copies/en/test.jsonl $EN/averitec_test.jsonl \
        data/processed/ar/xfact_ar_test.jsonl --out reports/tables
  else
    echo "[run_all] skip evaluation tables (no results/*/preds_*.jsonl yet)"
  fi
  step "R8 tables and figures" src.eval.figures
fi

echo "[run_all] done: $STAGES"
