"""Static HTML snapshot of demo result cards, for slides and the report (R10).

What it does
    Reads a file written by ``python -m src.demo.app --run-examples`` (the model's
    verdict, confidence and note for each of the 6 illustrative examples), adds the
    examples' evidence back, and renders the same cards the app shows
    (``src.demo.app.format_results``) into one HTML page. Notes pass through
    ``clean_note`` again, so fixes made after the run (e.g. the R10 "الCLAIM" fix)
    appear as the current code would produce them. Open the page in a browser (or a
    headless one) to take a screenshot. No model is run.

Inputs
    --examples results/R6/examples_rule.json   output of --run-examples
    --titles   titles of the examples to show (default: all)

Outputs
    --out reports/figures/demo_cards.html

How to run (repo root)
    python -m src.demo.snapshot --examples results/R6/examples_rule.json \\
        --titles "New metro station (AR)" "Metro stopped by floods (mixed)" --out demo_cards.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.demo.app import ILLUSTRATIVE, format_results
from src.demo.examples import EXAMPLES
from src.explain.generate import clean_note


def cards_html(report: dict, titles: list[str] | None = None) -> str:
    """One HTML page with a result card per claim of the chosen examples."""
    by_title = {ex["title"]: ex for ex in EXAMPLES}
    parts = []
    for row in report["examples"]:
        if titles and row["title"] not in titles:
            continue
        evidence = [{"text": t} for t in by_title[row["title"]]["evidence"]]
        results = [{**c, "explanation": clean_note(c["explanation"], len(evidence)),
                    "evidence": evidence} for c in row["claims"]]
        parts.append(format_results(results, f"{row['title']} · {ILLUSTRATIVE}"))
    method = report.get("verdict_method")
    if method is None and "components" in report:   # R6 files predate verdict_method
        method = ("trained verdict head (H4, twin loss)"
                  if report["components"].get("verdict (H3)") == "real" else
                  "evidence rule (strongest entail vs strongest contradict snippet)")
    footer = f'<p style="color:#57606a;font-size:0.8em">Verdict by: {method}</p>' if method else ""
    return ('<!doctype html><html><head><meta charset="utf-8"><title>TwinCheck demo cards</title>'
            '</head><body style="font-family:Segoe UI,Arial,sans-serif;max-width:720px;'
            'margin:16px auto;background:#fff">' + "".join(parts) + footer + "</body></html>")


def main() -> None:
    """CLI: write the snapshot page."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--examples", default="results/R6/examples_rule.json")
    ap.add_argument("--titles", nargs="*")
    ap.add_argument("--out", default="reports/figures/demo_cards.html")
    args = ap.parse_args()
    report = json.loads(Path(args.examples).read_text(encoding="utf-8"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(cards_html(report, args.titles), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
