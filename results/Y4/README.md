# Y4 · Arabic and mixed twins (Youssef)

Code: `src/copies/arabic_twins.py` (tests in `tests/copies/test_arabic_twins.py`).
Run: `notebooks/Y4_Y5_Arabic_Colab.ipynb`, Colab T4, Qwen3-8B 4-bit, greedy, seed 42, 158 minutes.

## What was built
200 English test sources (seed-42 sample of the 281 test sources that have a `_B` twin).
For each source four twins, in the shared format, in `data/copies/ar/test.jsonl`:

| Twin | Claim | Evidence | Gold | Kept | Dropped |
|---|---|---|---|---|---|
| `_A` | Arabic | Arabic | supported | 147 | 53 |
| `_M` | Arabic | English (original) | supported | 192 | 8 |
| `_B_A` | Arabic | Arabic version of the `_B` evidence swap | contradicted | 145 | 55 |
| `_C_A` | Arabic | Arabic version of the `_C` unrelated evidence | unverified | 158 | 42 |
| **Total** | | | | **642** | 158 |

- Name glossary: 3,044 spaCy names (persons, places, organisations, plus the names swapped in by `_B`)
  translated once (`data/translate/glossary_en_ar.csv`); every later translation is told to reuse them.
- 3,670 distinct texts translated; 164 (4.5%) still failed the automatic check after one retry
  (wrong script, mostly English left, or empty). A twin is dropped if any of its texts failed, so
  twins with many snippets (`_A`, `_B_A`) lose more. `_M` only needs the claim translated.
- Resumable cache: `data/copies/ar/translation_cache.jsonl`. Summary: `report.json`.

## Hand check (`check_50.csv`)
50 random translations (25 claims, 25 snippets), pre-checked by an LLM and reviewed by Youssef. Rule: `ok = no`
when an error could change whether the claim is true or false (a name, place, number, date, negation or key
term); leftover English words or odd style alone stay `yes`.

| | Correct | Wrong |
|---|---|---|
| Claims | 13 / 25 | 12 |
| Snippets | 14 / 25 | 11 |
| **All** | **27 / 50 (54%)** | 23 |

Main errors: names misspelled (Walker as وايلر, Rhode Island, North Carolina), key terms mistranslated
(Senate, earmarks, Affordable Care Act, IV bags as paper bags), facts changed (ninety days to thirty,
Hurricane Maria to Irma, "only 10%" dropped), and a few garbled sentences. The automatic script check
catches none of these. So the twins test the head on noisy translations, which is part of the Arabic gap
(`results/Y6/README.md`).
