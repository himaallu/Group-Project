# src/demo: Gradio demo

| File | Owner | What it does |
|---|---|---|
| `app.py` | Rathi (R6, R10) | web app around `src.pipeline.check`; `--share` for a public link, `--run-examples` to record the examples |
| `examples.py` | Rathi (R6) | six made-up UAE examples (2 per verdict, 2 Arabic, 1 mixed), shown as "Illustrative example" |
| `snapshot.py` | Rathi (R10) | renders a `--run-examples` file as result cards (HTML) for slides and the report |

Run: `python -m src.retrieval.build_index download && python -m src.demo.app`. Colab: `notebooks/R10_demo_colab.ipynb`. Tests: `tests/demo/`.

Buttons: 3 twin examples (one claim, its Arabic twin and an unrelated-evidence twin; tested with the trained head) and the 6 ready examples. A banner above each result names the verdict step; the rule stand-in is marked "Rule stand-in, not the trained model".

Each card shows the verdict, a confidence bar, the note (citations [1]–[3] link to their source) and the numbered evidence. Arabic claims, notes and snippets are right-to-left.

Verdict: the trained head `models/verdict_twin_s42` by default; `TWINCHECK_VERDICT=none` uses the simple rule (max entail vs max contradict), which gets 5 of 6 examples right against 2 of 6 for the head (`results/R6/README.md`). The footer says which one is running.
