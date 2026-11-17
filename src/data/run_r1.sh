#!/usr/bin/env bash
# R1 · English data pipeline (Rathi). Reproduces data/processed/en/ from scratch.
#
# What it does: download MultiFC + AVeriTeC, convert both to the shared format,
#   clean + split them (src/data/clean.py), run the unit tests.
# Inputs: internet access; data/label_map.csv.
# Outputs: data/processed/en/{multifc,averitec}_{train,val,test,nosnippets}.jsonl,
#   results/R1/clean_report_{multifc,averitec}.json
# How to run (from the repo root, or on Colab inside the Drive folder):
#   bash src/data/run_r1.sh            # full run (~5 min, no GPU)
#   LIMIT=10 bash src/data/run_r1.sh   # 10-row test run into data/processed/en/test10/
set -euo pipefail

OUT=data/processed/en
REPORTS=results/R1
LIMIT_ARG=""
if [[ -n "${LIMIT:-}" ]]; then
  OUT=data/processed/en/test10
  REPORTS=$OUT
  LIMIT_ARG="--limit $LIMIT"
fi

python -m src.data.download_en
# $LIMIT_ARG is left unquoted on purpose: empty -> no argument.
python -m src.data.load_multifc $LIMIT_ARG --out "$OUT/staging/multifc_all.jsonl"
python -m src.data.load_averitec $LIMIT_ARG --out "$OUT/staging/averitec_all.jsonl"
python -m src.data.clean --input "$OUT/staging/multifc_all.jsonl" --name multifc \
  --out-dir "$OUT" --split own --report "$REPORTS/clean_report_multifc.json"
python -m src.data.clean --input "$OUT/staging/averitec_all.jsonl" --name averitec \
  --out-dir "$OUT" --split keep --report "$REPORTS/clean_report_averitec.json"
python -m pytest tests/data -q
