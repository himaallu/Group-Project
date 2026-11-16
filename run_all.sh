#!/usr/bin/env bash
# Reproduces the main tables end to end. Owner: Rathi (R7). Stub created in H1.
set -euo pipefail

echo "[run_all] unit tests"
pytest -q

echo "[run_all] evaluation tables"
python -m src.eval.report \
  --gold data/processed/en/multifc_test.jsonl data/copies/en/test.jsonl \
  --out reports/tables
