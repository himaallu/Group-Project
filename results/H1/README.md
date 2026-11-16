# H1 · Repo + evaluation harness (Himasri)

1. **What ran:** repo layout (§6), `src/eval/loaders.py` (format checker), `src/eval/metrics.py` (all §5 metrics), `src/eval/report.py` (preds → CSV + LaTeX), `src/eval/smoke_test.py`; unit tests `python -m pytest -q`.
2. **Settings:** ECE 15 equal-width bins (confidence = prob of predicted label); paired bootstrap 1,000 resamples by `base_id`, seed 42, 95% percentile CI; claim match BGE-M3 sim ≥ 0.8; recall@5 = at least one own snippet in top 5.
3. **W&B:** none (no training in H1); project `twincheck` created for later runs.
4. **Key numbers:** 45/45 tests pass; fake fixture gives macro-F1 0.694, ECE 0.10, ESR 0.80, unrelated 0.50, paraphrase 0.75, cross-lingual 0.75, ALR 0.50 (all as expected).
5. **Big files:** none; fixtures in `tests/fixtures/`, smoke-test output in `results/H1/smoke/` (gitignored).
