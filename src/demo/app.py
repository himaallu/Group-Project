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
    Optional image (F11, src/demo/vision.py): an uploaded picture gets a separate card
    with the F5 detector's P(AI); the claim verdict does not change.

Inputs
    Typed in the browser. Models load on first use (GPU recommended).

Outputs
    A local web page; ``--share`` also prints a public gradio.live link (72 h).

How to run (repo root)
    python -m src.demo.app            # http://127.0.0.1:7860
    python -m src.demo.app --share    # in Colab
    python -m src.demo.app --run-examples results/R6/examples.json   # record the 6 examples
    python -m src.demo.app --share     # image card uses models/F5/vit_b16_s42.pt (git lfs pull)
"""

from __future__ import annotations

import argparse
import html
import io
import re
from pathlib import Path

from src.demo import vision
from src.demo.examples import EXAMPLES, TWIN_EXAMPLES
from src.demo.image_examples import CREDIT, load_examples
from src.demo.vision_card import image_note, summary_card

VERDICT_STYLE = {   # label -> (shown text, colour)
    "supported": ("Supported", "#1a7f37"),
    "contradicted": ("Contradicted", "#cf222e"),
    "unverified": ("Unverified", "#9a6700"),
}
ILLUSTRATIVE = "Illustrative example (made up, not real news)"
OFFLINE, WEB = "Offline library", "Live web search"
SOURCE = {"default": OFFLINE}         # --evidence-source sets the start value


def parse_evidence(text: str) -> list[str] | None:
    """One snippet per non-empty line; None (search the offline library) if empty."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines or None


FIRST_LETTER = re.compile(r"[A-Za-z\u0600-\u06FF]")   # first Latin or Arabic letter
CITATION = re.compile(r"\[(\d+)\]")


def text_dir(text: str) -> str:
    """'rtl' if the first letter of the text is Arabic, else 'ltr'.

    This is the "first strong character" rule that the browser uses for dir="auto",
    made explicit so the alignment can follow it too: a mixed message that starts
    in Arabic reads right-to-left, an English note that quotes an Arabic name does not.
    """
    m = FIRST_LETTER.search(text or "")
    return "rtl" if m and not m.group(0).isascii() else "ltr"


def dir_attrs(text: str) -> str:
    """HTML attributes that set direction and alignment for ``text``."""
    d = text_dir(text)
    return f'dir="{d}" style="text-align:{"right" if d == "rtl" else "left"}"'


def link_citations(note: str, evidence: list[dict]) -> str:
    """Escape the note and turn each [k] into a link to snippet k's source.

    A citation without a URL (e.g. pasted evidence) stays as bold [k]; a number
    outside the evidence list is left as plain text.
    """
    def repl(m: re.Match) -> str:
        k = int(m.group(1))
        if not 1 <= k <= len(evidence):
            return m.group(0)
        url = evidence[k - 1].get("url")
        if url:
            return (f'<a href="{html.escape(url)}" target="_blank" '
                    f'title="{html.escape(evidence[k - 1].get("domain") or url)}"><b>[{k}]</b></a>')
        return f"<b>[{k}]</b>"
    return CITATION.sub(repl, html.escape(note))


def confidence_bar(pct: int, colour: str) -> str:
    """A thin bar for the confidence, with the number as text for screen readers."""
    return (f'<div role="progressbar" aria-valuenow="{pct}" aria-valuemin="0" '
            f'aria-valuemax="100" aria-label="confidence {pct}%" '
            f'style="background:#eaeef2;border-radius:4px;height:8px">'
            f'<div style="background:{colour};width:{pct}%;height:8px;border-radius:4px">'
            f'</div></div>')


def claim_card(result: dict) -> str:
    """HTML card for one claim's result from ``check``.

    Shows the claim, the verdict badge, a confidence bar, the note with clickable
    citations, and the numbered evidence with source links. Arabic text is shown
    right-to-left (claim, note and each snippet separately, since mixed messages
    can have an Arabic claim with English evidence).
    """
    label, colour = VERDICT_STYLE.get(result["label"], (result["label"], "#57606a"))
    pct = round(100 * float(result["confidence"]))
    shown = result.get("evidence", [])[:3]
    items = []
    for i, ev in enumerate(shown, start=1):
        text = html.escape(ev["text"][:300])
        source = (f' <a href="{html.escape(ev["url"])}" target="_blank">'
                  f'{html.escape(ev.get("domain") or "source")}</a>' if ev.get("url") else "")
        items.append(f'<li {dir_attrs(ev["text"])}><b>[{i}]</b> {text}{source}</li>')
    evidence = "".join(items) or "<li>No evidence found.</li>"
    return (
        '<div style="border:1px solid #d0d7de;border-radius:8px;padding:12px;margin:8px 0">'
        f'<div {dir_attrs(result["claim"])}><b>{html.escape(result["claim"])}</b></div>'
        f'<div style="margin:6px 0"><span style="background:{colour};color:#fff;'
        f'padding:2px 8px;border-radius:12px">{label}</span> confidence {pct}%</div>'
        f'{confidence_bar(pct, colour)}'
        f'<p {dir_attrs(result["explanation"])}>'
        f'{link_citations(result["explanation"], shown)}</p>'
        f'<ol style="list-style:none;padding-left:0;font-size:0.9em">{evidence}</ol></div>')


def verdict_method() -> str:
    """Which verdict step is running, in words for the page footer."""
    from src.pipeline import component_status

    if component_status()["verdict (H3)"] == "real":
        return "trained verdict head (H4, twin loss)"
    return "evidence rule (strongest entail vs strongest contradict snippet)"


def method_banner(method: str) -> str:
    """A line above the results naming the verdict step; the rule stand-in is marked."""
    if method.startswith("trained"):
        return (f'<div style="background:#ddf4ff;color:#1f2328;padding:6px 10px;border-radius:6px;'
                f'margin:4px 0">Verdict by: {html.escape(method)}</div>')
    return ('<div style="background:#ffebe9;color:#1f2328;padding:6px 10px;border-radius:6px;margin:4px 0">'
            f'<b>Rule stand-in, not the trained model.</b> Verdict by: {html.escape(method)}'
            '</div>')


def format_results(results: list[dict], note: str = "", method: str = "") -> str:
    """HTML for all claims, with optional banners (e.g. "Illustrative example", verdict step)."""
    banner = (f'<div style="background:#fff8c5;color:#1f2328;padding:6px 10px;border-radius:6px">'
              f'{html.escape(note)}</div>') if note else ""
    banner += method_banner(method) if method else ""
    if not results:
        return banner + "<p>No check-worthy claim found in the message.</p>"
    return banner + "".join(claim_card(r) for r in results)


def open_image(image) -> tuple:
    """(PIL image, original file bytes) from the upload box (a file path) or a PIL image.

    The original bytes are kept for the provenance-tag check (F11); a PIL image has none.
    """
    if image is None or not isinstance(image, (str, bytes)) and not hasattr(image, "__fspath__"):
        return image, None
    from PIL import Image

    raw = Path(image).read_bytes()
    return Image.open(io.BytesIO(raw)).convert("RGB"), raw


def live_check(message: str, lang: str | None) -> tuple[list[dict], str]:
    """Check each claim against live web evidence (R10): (results, banner HTML).

    Claims are extracted as in ``check``; each claim gets its own search
    (src/retrieval/live.py) and is then checked with that evidence. If every search
    engine fails, that claim uses the offline library and the banner says so. Demo only.
    """
    from src.pipeline import check, detect_language, extract_claims
    from src.retrieval import live

    results, fallback = [], False
    for claim in extract_claims(message, lang or detect_language(message)):
        snippets = live.search(claim)
        if not snippets and live.last_error:            # every engine failed: use the offline library
            fallback = True
            results += check(claim, None, lang)
        else:
            results += check(claim, snippets, lang)
    note = ("Web search was unavailable, so evidence was taken from the offline library."
            if fallback else "Evidence retrieved from the web (fact-checking sites removed; top 5 by the reranker).")
    banner = f'<div style="font-size:0.85em;opacity:0.8;margin:4px 0">{note}</div>'
    return results, banner


def library_warning(evidence_text: str) -> str:
    """A warning when the evidence box is empty and the offline library is not loaded.

    Without the library and without pasted evidence there is nothing to read, so every
    claim would be Unverified; this says why instead of failing silently.
    """
    from src.pipeline import component_status

    if parse_evidence(evidence_text) is not None or component_status()["get_evidence (R5)"] == "real":
        return ""
    return ('<div style="background:#ffebe9;color:#1f2328;padding:6px 10px;border-radius:6px;'
            'margin:4px 0"><b>Offline evidence library not loaded</b>, so there is no evidence to '
            'check and the claim can only be Unverified. Paste evidence (one snippet per line) or '
            'run <code>python -m src.retrieval.build_index download</code>.</div>')


def run_check(message: str, evidence_text: str, lang: str, note: str = "", image=None,
              image_example: dict | None = None, source: str = OFFLINE) -> str:
    """Check a message and return the result HTML (what the Check button calls).

    With an ``image`` (F11), its card, a final result for text and image together, and a
    note follow the claim cards. The image is never passed to ``check``, so the claim
    verdict is the same with or without it. ``image_example`` (a built-in case) adds its
    description, its cached F3 input and its known label. ``source`` (when no evidence is
    pasted): the offline library or live web search (R10, demo only).
    """
    from src.pipeline import check

    image, raw = open_image(image)
    lang = None if lang in (None, "", "auto") else lang
    if image is not None and not (message or "").strip():      # image only: no claim to check
        card, _ = vision.image_check(image, raw=raw)
        return (card + '<p style="opacity:0.8">Paste a message as well to check its claim; '
                       'the claim verdict comes from the text evidence only.</p>')
    if source == WEB and parse_evidence(evidence_text) is None:
        results, banner = live_check(message, lang)
    else:
        results = check(message, parse_evidence(evidence_text), lang)
        banner = library_warning(evidence_text)
    out = banner + format_results(results, note, verdict_method())
    if image is not None:
        ex = image_example or {}
        card, image_result = vision.image_check(image, blob=ex.get("blob"), truth=ex.get("truth"), raw=raw)
        case = (f'<div style="margin:8px 0 0;opacity:0.85"><b>Case:</b> {html.escape(ex["case"])}</div>'
                if ex.get("case") else "")
        out += case + card + summary_card(results, image_result) + image_note()
    return out


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
                image = gr.Image(type="filepath", label="Image (optional)", height=240)
                message = gr.Textbox(label="Message", lines=4,
                                     placeholder="Paste a message (English, Arabic or mixed)")
                evidence = gr.Textbox(label="Evidence (optional, one snippet per line)", lines=3)
                lang = gr.Dropdown(["auto", "en", "ar", "mixed"], value="auto", label="Language")
                button = gr.Button("Check", variant="primary")
            with gr.Column():
                output = gr.HTML()
        button.click(lambda i, m, e, l: run_check(m, e, l, image=i, source=SOURCE["default"]),
                     [image, message, evidence, lang], output)

        # Twin examples first: one claim and two of its twins, the core idea of the project.
        for heading, group in ((f"### Twin examples — {ILLUSTRATIVE}", TWIN_EXAMPLES),
                               (f"### Ready examples — {ILLUSTRATIVE}", EXAMPLES)):
            gr.Markdown(heading)
            with gr.Row():
                for ex in group:
                    b = gr.Button(f"{ex['title']}", size="sm")
                    b.click(lambda ex=ex: (ex["message"], "\n".join(ex["evidence"]), ex["lang"], None,
                                           run_check(ex["message"], "\n".join(ex["evidence"]),
                                                     ex["lang"], ILLUSTRATIVE)),
                            None, [message, evidence, lang, image, output])

        # Text + image cases (F11): known-label images from the repo, one per case in the brief.
        cases = load_examples()
        if cases:
            gr.Markdown(f"### Text + image cases — {ILLUSTRATIVE}. Each card shows the image's "
                        f"known label and whether the detector got it right.")
            with gr.Row():
                for ex in cases:
                    b = gr.Button(ex["title"], size="sm")
                    b.click(lambda ex=ex: _case_example(ex), None,
                            [message, evidence, lang, image, output])
            gr.Markdown(f"<small>{html.escape(CREDIT)}</small>")

        # More image examples from --demo-images (pictures + examples.json, not in git).
        image_examples = vision.load_image_examples()
        if image_examples:
            gr.Markdown(f"### Image examples — {ILLUSTRATIVE}")
            with gr.Row():
                for ex in image_examples:
                    b = gr.Button(ex["title"], size="sm")
                    b.click(lambda ex=ex: _image_example(ex), None,
                            [message, evidence, lang, image, output])
        status = ", ".join(f"{k}: {v}" for k, v in component_status().items())
        gr.Markdown(f"<small>Verdict by: {html.escape(verdict_method())}. "
                    f"Components: {html.escape(status)}, {html.escape(vision.status())}</small>")
    return app


def _case_example(ex: dict) -> tuple:
    """Fill the inputs for a built-in text + image case and run the check (F11)."""
    evidence = "\n".join(ex["evidence"])
    return ex["message"], evidence, ex["lang"], ex["image"], run_check(
        ex["message"], evidence, ex["lang"], ILLUSTRATIVE, image=ex["image"], image_example=ex)


def _image_example(ex: dict) -> tuple:
    """Fill the inputs for an image example and run the check (F11)."""
    from PIL import Image

    picture = Image.open(ex["image_path"]).convert("RGB")
    evidence = "\n".join(ex["evidence"])
    lang = ex.get("lang", "auto")
    return ex["message"], evidence, lang, picture, run_check(ex["message"], evidence, lang,
                                                             ILLUSTRATIVE, image=picture)


def run_examples(path: str) -> list[dict]:
    """Run the 3 twin examples and the 6 ready examples through ``check``; save the results.

    Returns:
        One row per example: title, lang, expected, and per claim the label,
        confidence and note. Also written to ``path`` (JSON).
    """
    import json
    from pathlib import Path

    from src.pipeline import check, component_status

    rows = []
    for ex in TWIN_EXAMPLES + EXAMPLES:
        results = check(ex["message"], ex["evidence"], ex["lang"])
        rows.append({"title": ex["title"], "lang": ex["lang"], "expected": ex["expected"],
                     "claims": [{"claim": r["claim"], "label": r["label"],
                                 "confidence": round(float(r["confidence"]), 3),
                                 "explanation": r["explanation"]} for r in results]})
    report = {"components": component_status(), "verdict_method": verdict_method(),
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
    ap.add_argument("--vision-ckpt", default="models/F5/vit_b16_s42.pt",
                    help="F5 detector checkpoint (image card, F11; Git LFS)")
    ap.add_argument("--jpeg-pool", default="models/F11/jpeg_pool.json",
                    help="F3 JPEG settings (src.vision.infer --pool-from-cache)")
    ap.add_argument("--demo-images", help="folder with examples.json + images for the image buttons")
    ap.add_argument("--external-detector", default="Organika/sdxl-detector",
                    help="second-opinion AI-image detector from the Hub, demo only ('none' = off)")
    ap.add_argument("--evidence-source", choices=["offline", "web"], default="offline",
                    help="start value of the evidence source (web = live DuckDuckGo search, R10)")
    args = ap.parse_args()
    SOURCE["default"] = WEB if args.evidence_source == "web" else OFFLINE
    vision.configure(args.vision_ckpt, args.jpeg_pool, args.demo_images, args.external_detector)
    if args.run_examples:
        for row in run_examples(args.run_examples):
            got = ", ".join(c["label"] for c in row["claims"]) or "no claim"
            print(f"{row['title']}: expected {row['expected']}, got {got}")
        return
    build_app().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
