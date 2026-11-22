# R10 · Demo polish, video script, slides, MAAI901 review (Rathi)

- **What ran:** `src/demo/app.py` polish: Arabic claims, notes and snippets are right-to-left (first-letter rule, each part separately); a confidence bar with screen-reader text; citations [1]–[3] in the note link to their source; the footer names the verdict step (trained head or evidence rule). `src/explain/generate.py`: Arabic notes call the claim "الادعاء", and `clean_note` replaces any English "claim" left in an Arabic note (the R6 "الCLAIM" glitch). `src/demo/snapshot.py` renders saved results as cards for slides. 3 slides (`reports/slides/R10_rathi_slides.pptx`, built by `make_r10_slides.py`). Video script (`notes/R10_demo_video_script.md`) and Colab notebook (`notebooks/R10_demo_colab.ipynb`). Technical review of the MAAI901 skeleton (`notes/R10_maai901_review.md`).
- **Settings:** no model was retrained and no new GPU run, so there is no W&B run. The slide card (`reports/slides/demo_cards.png`) is the R6 rule run (`results/R6/examples_rule.json`) re-rendered with the current code. Every slide number is read from `reports/tables/` and `results/`.
- **Key numbers:** demo on R6 examples: evidence rule 5 of 6 expected verdicts, trained head 2 of 6. Tests: 152 pass, 8 skipped (local run; torch and lightgbm tests skipped here, they run in Colab).
- **Still to do:** you record the 2–3 min video with the R10 notebook (set `VERDICT` to what the group agrees). Second review pass once Feba pushes the full MAAI901 draft (F7, Fri 21:00).
- **Files:** `src/demo/{app,snapshot}.py`, `src/explain/generate.py`, `tests/demo/test_r10.py`, `reports/slides/`, `notebooks/R10_demo_colab.ipynb`, `notes/R10*.md`.
- **Update, Sun 22 Nov:**
  - Demo has 3 twin example buttons (original EN → contradicted, Arabic twin → contradicted, unrelated evidence → unverified; Himasri tested them with the trained head at 83% / 83% / 97%).
  - A banner above every result names the verdict step and marks the rule stand-in in red.
  - The R10 notebook now defaults to the trained head and a FAST mode (~5 min to the link).
  - The video script is built around the twins.
  - Rule-run results from today's Colab: `results/R10/examples_rule.json`, 5 of 6.
- **Colab run with the trained head (FAST mode), Sun 22 Nov:** `results/R10/examples_head_fast.json`, every component real.
  - Twins as expected: original EN → contradicted 0.829, Arabic twin → contradicted 0.829, unrelated evidence → unverified 0.971. Arabic and English give the same answer, and unrelated evidence gives unverified.
  - Ready examples: 2 of 6 right (both contradicted ones). The head calls the supported and unverified examples contradicted or supported, which is its known "supported" weakness. The demo video therefore uses the twin buttons.
