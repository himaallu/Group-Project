# Y2 · Arabic NLI data (Youssef)

## Training pairs: `data/nli/ar_train.jsonl` (for Rathi's R4)
2,658 pairs, balanced: 886 entail / 886 contradict / 886 neutral. Same fields as `gold_val.jsonl` plus `origin`.

| origin | pairs | how |
|---|---|---|
| `xfact_weak` | 1,818 | X-FACT Arabic train (cleaned), weak labels: supported + best-overlap snippet = entail, contradicted + snippet = contradict, snippet of another claim = neutral (`src/nli/build_ar_pairs.py`) |
| `multifc_translated` | 840 | MultiFC supported claims + best snippet (overlap >= 0.5), translated EN to MSA with Qwen3-8B 4-bit, greedy (`src/translate/translate_pairs.py`) |

- X-FACT Arabic has only 46 entail pairs (true claims are rare), so entail is topped up with translations.
- 954 pairs were translated; 114 (12%) were dropped because the model drifted into Cyrillic/Chinese or left the text mostly in English. All three classes were then cut to 886, the entail count.
- Known translation issues (for the error analysis): some place names are mistranslated (e.g. "South Carolina" became "سيريليا الجنوبية"), and claims starting "Says ..." are kept literally ("تقول ...").
- No gold claim appears in training (the 150 gold-pool claims are excluded; checked: 0 overlap).

## Gold pairs: 150 Arabic rows in `data/nli/gold_val.jsonl`
- Claims: X-FACT Arabic val (93) + 57 held-out train claims (10 supported / 7 unverified / 40 contradicted, seed 42); sheet from `src/nli/sample_gold.py`.
- Labels: 36 entail / 22 contradict / 92 neutral (`gold_sheet_ar_labelled.csv`; notes: 9 unsure, 9 junk, 8 factcheck, 6 partial).
- Method: labels drafted with an LLM following `labelling_guide`, then checked by Youssef; snippet-only rule, no outside knowledge.

## Re-run
    python -m src.nli.build_ar_pairs
    python -m src.translate.translate_pairs --n 954     # GPU, ~45 min on a T4
    python -m src.translate.translate_pairs --merge
    python -m src.nli.sheet_to_jsonl results/Y2/gold_sheet_ar_labelled.csv --out data/nli/gold_val.jsonl
