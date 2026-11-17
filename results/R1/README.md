# R1 · English data (Rathi)

- **What ran:** `bash src/data/run_r1.sh` = download MultiFC (authors' release with snippets) + AVeriTeC (HF `pminervini/averitec`) → `load_multifc.py`, `load_averitec.py` (shared format, `data/label_map.csv`) → `clean.py` (5 rules) → `tests/data` (8 tests pass); every output passes `src/eval/loaders.py --check` (H1).
- **Settings:** MultiFC train+dev only (official test labels never released), own 80/10/10 split stratified by label, seed 42; AVeriTeC official dev = our test, 10% of train = val (stratified, seed 42); fact-check sites + fact-check URL paths dropped; snippets with >90% of claim words dropped; ≤10 snippets per claim. No training, so no W&B run.
- **Key numbers:** MultiFC 16,847 / 2,104 / 2,106 claims (train/val/test), 7.3 snippets per claim; AVeriTeC 2,309 / 252 / 429, ~2.5 snippets per claim. Rule 1 removed 25,993 MultiFC snippets (13.2%), rule 2 removed 16,769 (8.5%). Full table: `counts.csv`; per-rule counts: `clean_report_*.json`.
- **Samples:** `samples/<dataset>_<split>_10rows.jsonl` (first 10 rows of each hand-off file).
- **Big files:** Drive `TwinCheck/data/processed/en/{multifc,averitec}_{train,val,test,nosnippets}.jsonl` (~75 MB) and `TwinCheck/data/raw/` (not in git).
