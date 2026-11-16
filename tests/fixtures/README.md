# Test fixtures (fake data, no real claims)

`fake_gold.jsonl` (shared record format) and `fake_preds.jsonl` (predictions format):
10 originals `multifc_000001`–`010` plus 17 twins. Every prediction puts 0.8 on the
predicted label and 0.1 on each of the other two.

Expected metrics (checked in `tests/eval/test_metrics.py`):

| Metric | Rows | Expected |
|---|---|---|
| Macro-F1 (originals) | 7/10 correct | 0.6944 (F1 per class 0.75 / 0.667 / 0.667) |
| ECE (originals) | conf 0.8, accuracy 0.7 | 0.10 |
| ESR | 5 `_B` (incl. Arabic `003_B_A`), 4 contradicted | 0.80 |
| Unrelated rate | 2 `_C`, 1 unverified | 0.50 |
| Paraphrase consistency | 4 `_P`, `004_P` differs from its original | 0.75 |
| Cross-lingual (A / M / both) | `002_M` differs | 1.00 / 0.50 / 0.75 |
| ALR | 2 `_V`, `006_V` changes | 0.50 |
