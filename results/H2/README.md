# H2 · English NLI gold pairs (Himasri)

1. **What ran:** `python -m src.nli.sample_gold data/processed/en/multifc_val.jsonl --n 150` → hand-labelled `gold_sheet_en.csv` → `python -m src.nli.sheet_to_jsonl results/H2/gold_sheet_en.csv --out data/nli/gold_val.jsonl`.
2. **Settings:** MultiFC val only, 50 claims per verdict label, 1 snippet per claim (max word overlap), seed 42; labels entail / contradict / neutral per `notes/H2_labelling_guide.md`.
3. **W&B:** none (no training).
4. **Key numbers:** 150 pairs: 10 entail, 8 contradict, 132 neutral (88% neutral); 16 unsure, 13 partial, 3 junk, 2 leaked fact-check; 3 labels changed e → n on review.
5. **Big files:** `data/nli/gold_val.jsonl` in Drive (`TwinCheck/data/nli/`); this folder holds the labelled sheet (`gold_sheet_en.csv`, notes column included).
