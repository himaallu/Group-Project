# Y6 · Arabic error analysis (Youssef)

## 1. Arabic twins: 20 Arabic-only failures (`arabic_twin_failures.csv`)
System twin_noent, seed 42. Over seeds, about 85% of wrong Arabic or mixed verdicts are also wrong for
the English twin (`results/Y5/diagnostics.csv`). For seed 42 that leaves 57 errors where English is
right and Arabic is wrong (`_B_A` 25, `_M` 17, `_A` 15). We read 20 of them, claim and evidence side by side:

| Cause | n | Example |
|---|---|---|
| Translation changed the meaning | 6 | "Swaziland has the highest rate of HIV infection" became "Sudan has the highest rate of *no* HIV infection" |
| Name spelled differently | 3 | "Bush" as بوس in the claim, بوش in the evidence; Reagan spelled two ways |
| English or foreign letters left in | 3 | "أ Bankrupt", "الديمOCRATية", Hebrew letters in one word |
| Swapped evidence read as unrelated | 3 | Arabic `_B_A` predicted unverified, English `_B` contradicted |
| No visible translation error | 3 | good translation, still predicted contradicted (head bias) |
| Arabic claim, English evidence (`_M`) | 2 | correct claim, head does not link it to the English sentence |

12 of the 20 (60%) are translation problems, so part of the Arabic gap is in the test data, not the
model. A stricter check (name spelling against the glossary, any Latin or Hebrew fragment) would
catch about half of them.

## 2. X-FACT Arabic test: 36 failures (`xfact_ar_failures.csv`)
System twin, seed 42, real Misbar claims.

| Cause | n |
|---|---|
| Rumour restated in the snippets (snippet repeats the false claim, so NLI says entail) | 15 |
| Missing evidence: claim is about a photo or video | 5 |
| Broken encoding in the snippets (Thai-range mojibake) | 4 |
| Model bias to contradicted | 4 |
| Missing evidence: no snippet text (`<DUMMY_EVIDENCE>`) | 3 |
| Conflicting snippets | 2 |
| Name or entity mismatch | 2 |
| Cross-lingual: English-only evidence | 1 |

Data issue: X-FACT Arabic snippets contain mojibake (train 51, val 4, test 22) and `<DUMMY_EVIDENCE>`
(train 43, val 3, test 7); 12 test claims have only bad evidence. The biggest cause, rumours restated
by search results, is a known weakness of search-snippet evidence: the snippet *mentions* the claim
without *confirming* it, and NLI cannot tell the difference.
