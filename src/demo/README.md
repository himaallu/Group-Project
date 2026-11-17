# src/demo: Gradio demo

| File | Owner | What it does |
|---|---|---|
| `app.py` | Rathi (R6) | web app around `src.pipeline.check`; `--share` for a public link, `--run-examples` to record the examples |
| `examples.py` | Rathi (R6) | six made-up UAE examples (2 per verdict, 2 Arabic, 1 mixed), shown as "Illustrative example" |

Run: `python -m src.retrieval.build_index download && python -m src.demo.app`. Tests: `tests/demo/`.
