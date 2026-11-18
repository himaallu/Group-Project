# Twin hand-check guide (Himasri + Feba, Wed 08:00)

Each of you rates `results/H4/handcheck_<you>.csv` (75 test twins: 25 `_P`, 25 `_B`, 25 `_C`; 30 rows are on both sheets, you are not told which). Fill `valid` with **y** or **n**, and `note` when it is n or you are unsure.

- **y** = the twin changes exactly one thing and `expected_label` is the right verdict for it.
- `_P` (paraphrase): same meaning, same names / dates / numbers as `orig_claim` → y. Meaning changed, a fact added or dropped → n. A near-identical copy (one word changed) is still **y** (note `minimal`).
- `_B` (evidence swap): `twin_evidence` shows the changed snippets; the claim's place / person / year is replaced by a different real one, so the evidence now contradicts the claim → y. Swap that does not contradict the claim (e.g. first name replaced by an unrelated surname, entity not the claim's), broken grammar that changes meaning, or the original entity still present → n.
- `_C` (unrelated evidence): the new snippets say nothing about the claim, so "unverified" is right → y. If any new snippet actually supports or contradicts the claim → n.

Then: `python -m src.copies.handcheck_kappa results/H4/handcheck_himasri.csv results/H4/handcheck_feba.csv` → kappa (target ≥ 0.7) and valid rate per twin type in `results/H4/handcheck_summary.json`.
