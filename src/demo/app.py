"""TwinCheck demo: a Gradio web app around ``src.pipeline.check`` (R6).

What it does
    The user pastes a message (English, Arabic or mixed) and, optionally, their own
    evidence (one snippet per line). The app calls ``check(message, evidence, lang)``
    (Himasri's pipeline: claims -> evidence -> NLI -> verdict -> calibration -> cited
    note) and shows one card per claim: the verdict, a confidence bar, the cited note
    and the numbered evidence with links. With no evidence given, the offline library
    is searched (R5; ``python -m src.retrieval.build_index download`` first).
    Six ready examples (src/demo/examples.py) are made up and shown as
    "Illustrative example". Arabic text is shown right-to-left.
    The verdict never uses any "is the image AI-made?" score (CLAUDE.md §3).

Inputs
    Typed in the browser. Models load on first use (GPU recommended).

Outputs
    A local web page; ``--share`` also prints a public gradio.live link (72 h).

How to run (repo root)
    python -m src.demo.app            # http://127.0.0.1:7860
    python -m src.demo.app --share    # in Colab
    python -m src.demo.app --run-examples results/R6/examples.json   # record the 6 examples
"""

from __future__ import annotations

import argparse
import html

from src.demo.examples import EXAMPLES

VERDICT_STYLE = {   # label -> (shown text, colour)
    "supported": ("Supported", "#1a7f37"),
    "contradicted": ("Contradicted", "#cf222e"),
    "unverified": ("Unverified", "#9a6700"),
}
ILLUSTRATIVE = "Illustrative example (made up, not real news)"


def parse_evidence(text: str) -> list[str] | None:
    """One snippet per non-empty line; None (search the offline library) if empty."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines or None


def claim_card(result: dict) -> str:
    """HTML card for one claim's result from ``check``."""
    label, colour = VERDICT_STYLE.get(result["label"], (result["label"], "#57606a"))
    pct = round(100 * float(result["confidence"]))
    items = []
    for i, ev in enumerate(result.get("evidence", [])[:3], start=1):
        text = html.escape(ev["text"][:300])
        source = (f' <a href="{html.escape(ev["url"])}" target="_blank">'
                  f'{html.escape(ev.get("domain") or "source")}</a>' if ev.get("url") else "")
        items.append(f'<li dir="auto"><b>[{i}]</b> {text}{source}</li>')
    evidence = "".join(items) or "<li>No evidence found.</li>"
    return (
        '<div style="border:1px solid #d0d7de;border-radius:8px;padding:12px;margin:8px 0">'
        f'<div dir="auto" style="font-weight:600">{html.escape(result["claim"])}</div>'
        f'<div style="margin:6px 0"><span style="background:{colour};color:#fff;'
        f'padding:2px 8px;border-radius:12px">{label}</span> confidence {pct}%</div>'
        f'<div style="background:#eaeef2;border-radius:4px;height:8px">'
        f'<div style="background:{colour};width:{pct}%;height:8px;border-radius:4px"></div></div>'
        f'<p dir="auto">{html.escape(result["explanation"])}</p>'
        f'<ol style="list-style:none;padding-left:0;font-size:0.9em">{evidence}</ol></div>')


def format_results(results: list[dict], note: str = "") -> str:
    """HTML for all claims, with an optional banner (e.g. "Illustrative example")."""
    banner = (f'<div style="background:#fff8c5;padding:6px 10px;border-radius:6px">'
              f'{html.escape(note)}</div>') if note else ""
    if not results:
        return banner + "<p>No check-worthy claim found in the message.</p>"
    return banner + "".join(claim_card(r) for r in results)


def run_check(message: str, evidence_text: str, lang: str, note: str = "") -> str:
    """Check a message and return the result HTML (what the Check button calls)."""
    from src.pipeline import check

    lang = None if lang in (None, "", "auto") else lang
    return format_results(check(message, parse_evidence(evidence_text), lang), note)


def build_app():
    """The Gradio Blocks app."""
    import gradio as gr
    from src.pipeline import component_status

    with gr.Blocks(title="TwinCheck") as app:
        gr.Markdown("# TwinCheck\nBilingual (Arabic–English) evidence-grounded fact-checking. "
                    "Each claim gets **Supported / Contradicted / Unverified**, a confidence "
                    "and a short note citing the evidence.")
        with gr.Row():
            with gr.Column():
                message = gr.Textbox(label="Message", lines=4,
                                     placeholder="Paste a message (English, Arabic or mixed)")
                evidence = gr.Textbox(label="Evidence (optional, one snippet per line; "
                                            "empty = search the offline library)", lines=4)
                lang = gr.Dropdown(["auto", "en", "ar", "mixed"], value="auto", label="Language")
                button = gr.Button("Check", variant="primary")
            with gr.Column():
                output = gr.HTML()
        button.click(lambda m, e, l: run_check(m, e, l), [message, evidence, lang], output)

        gr.Markdown(f"### Ready examples — {ILLUSTRATIVE}")
        with gr.Row():
            for ex in EXAMPLES:
                b = gr.Button(f"{ex['title']}", size="sm")
                b.click(lambda ex=ex: (ex["message"], "\n".join(ex["evidence"]), ex["lang"],
                                       run_check(ex["message"], "\n".join(ex["evidence"]),
                                                 ex["lang"], ILLUSTRATIVE)),
                        None, [message, evidence, lang, output])
        status = ", ".join(f"{k}: {v}" for k, v in component_status().items())
        gr.Markdown(f"<small>Components: {html.escape(status)}</small>")
    return app


def run_examples(path: str) -> list[dict]:
    """Run the six examples through ``check`` and save what the model says.

    Returns:
        One row per example: title, lang, expected, and per claim the label,
        confidence and note. Also written to ``path`` (JSON).
    """
    import json
    from pathlib import Path

    from src.pipeline import check, component_status

    rows = []
    for ex in EXAMPLES:
        results = check(ex["message"], ex["evidence"], ex["lang"])
        rows.append({"title": ex["title"], "lang": ex["lang"], "expected": ex["expected"],
                     "claims": [{"claim": r["claim"], "label": r["label"],
                                 "confidence": round(float(r["confidence"]), 3),
                                 "explanation": r["explanation"]} for r in results]})
    report = {"components": component_status(),
              "examples": rows}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def main() -> None:
    """CLI: launch the app (or only run the examples)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--run-examples", metavar="JSON",
                    help="run the 6 examples, save the results here and exit")
    args = ap.parse_args()
    if args.run_examples:
        for row in run_examples(args.run_examples):
            got = ", ".join(c["label"] for c in row["claims"]) or "no claim"
            print(f"{row['title']}: expected {row['expected']}, got {got}")
        return
    build_app().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
