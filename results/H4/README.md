# H4 (prep, done early) · Calibration, hand-check, pipeline interface, zero-shot NLI script (Himasri)

1. **What ran:** built `src/verdict/calibrate.py`, `src/copies/handcheck_sheet.py` + `handcheck_kappa.py`, `src/pipeline.py` (`check(message, evidence=None, lang=None)` with stand-ins), `src/nli/score_zeroshot.py`, `cohen_kappa` in `src/eval/metrics.py`; 28 new unit tests (90 total pass).
2. **Settings:** temperature by NLL grid search (T in 0.05–10), Unverified threshold by val macro-F1 (0 or 0.30–0.90); hand-check 75 twins per rater, 30 shared, balanced by twin type, seed 42; zero-shot model `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, max len 256.
3. **W&B:** none (no training).
4. **Key numbers:** zero-shot NLI on the 150 EN gold pairs: macro-F1 0.367, accuracy 0.513 — it calls 67 of 132 neutral pairs entail/contradict (`zeroshot_gold_check.json`); 10-row run: 66 pairs in 4 s on CPU.
5. **Big files:** none yet. Wednesday outputs go to `results/H4/handcheck_*.csv`, `handcheck_summary.json`, `models/verdict_calib.json`; zero-shot features to `features/nli_zeroshot/<split>.parquet` (run on Colab).
