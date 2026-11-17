# H4 prep (Himasri, Tue 17 Nov) — Wednesday code built early while waiting for R2/R3

- Calibration: `src/verdict/calibrate.py` fits temperature T (val NLL) then the Unverified threshold tau (val macro-F1); settings saved to `models/verdict_calib.json`, input is a val logits JSONL `{id, gold, logits[3]}`.
- Hand-check: `python -m src.copies.handcheck_sheet --twins data/copies/en/test.jsonl --originals data/processed/en/multifc_test.jsonl` makes both sheets; `python -m src.copies.handcheck_kappa <h.csv> <f.csv>` gives kappa (target >= 0.7) and valid rate per twin type.
- Pipeline: `src/pipeline.py` `check()` returns `{claim, language, label, confidence, probs, evidence, explanation}` per claim; teammates' modules plug in automatically (expected names: `src.claims.extract.extract_claims`, `src.retrieval.index.search`, `src.nli.score.score_pairs`, `src.verdict.model.predict_one`, `src.explain.generate.generate`); `python -m src.pipeline --status` shows what is still a stub.
- Zero-shot NLI: `src/nli/score_zeroshot.py` → `features/nli_zeroshot/<file>.parquet` (`id, snippet_idx, p_entail, p_contradict, p_neutral`); run the full splits on Colab GPU.
- Finding: zero-shot mDeBERTa gets macro-F1 0.367 on our gold pairs and labels half the neutral pairs as entail/contradict (topic overlap read as support), which motivates R4's fine-tuning and the twin tests.
