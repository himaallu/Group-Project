"""Rathi's 3 presentation slides (R10): data + twins, demo, repository.

What it does
    Builds a 16:9 PowerPoint file with three slides and speaker notes. Every number
    is read from files in reports/tables/ and results/ (none typed by hand), and the
    figures come from reports/figures/ and reports/slides/demo_cards.png
    (made by ``python -m src.demo.snapshot`` + a browser screenshot).
    Himasri copies the slides into the full deck (H8).

Inputs
    reports/tables/tab_data_counts.csv, tab_twins.csv, tab_retrieval.csv, tab_claims.csv
    results/R5/retrieval_eval.json (library size), results/R6/examples{,_rule}.json
    reports/figures/fig_twin_counts.png, reports/slides/demo_cards.png

Outputs
    reports/slides/R10_rathi_slides.pptx

How to run (repo root; needs python-pptx)
    python reports/slides/make_r10_slides.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

INK = RGBColor(0x1F, 0x23, 0x28)
MUTED = RGBColor(0x57, 0x60, 0x6A)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
BOX = RGBColor(0xF3, 0xF6, 0xFA)
OUT = Path("reports/slides/R10_rathi_slides.pptx")


def read_csv(path: str) -> list[dict]:
    """Rows of a CSV file as dicts."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def n(x) -> str:
    """Integer with thousands separators: 16847 -> '16,847'."""
    return f"{int(float(x)):,}"


def text_box(slide, x, y, w, h, lines, size=16, color=INK, bold_first=False, font=None):
    """Add a text box; ``lines`` is a list of strings (one paragraph each)."""
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold_first and i == 0
        if font:
            run.font.name = font
        p.space_after = Pt(4)
    return tf


def title(slide, text, sub):
    """Slide title and a one-line subtitle."""
    text_box(slide, 0.5, 0.3, 12.3, 0.7, [text], size=30, bold_first=True)
    text_box(slide, 0.5, 0.95, 12.3, 0.5, [sub], size=16, color=MUTED)


def table(slide, x, y, w, rows, col_w, size=13):
    """A simple table; first row is the header."""
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                   Inches(w), Inches(0.36 * len(rows)))
    for c, cw in enumerate(col_w):
        shape.table.columns[c].width = Inches(cw)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = shape.table.cell(r, c)
            cell.text = str(value)
            para = cell.text_frame.paragraphs[0]
            para.runs[0].font.size = Pt(size)
            para.runs[0].font.bold = r == 0
            para.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if r == 0 else INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT if r == 0 else (BOX if r % 2 else RGBColor(0xFF, 0xFF, 0xFF))
    return shape


def flow(slide, y, steps):
    """A row of boxes joined by arrows: [(heading, detail), ...]."""
    gap, x0 = 0.25, 0.5
    w = (12.3 - gap * (len(steps) - 1)) / len(steps)
    for i, (head, detail) in enumerate(steps):
        x = x0 + i * (w + gap)
        box = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(1.15))
        box.fill.solid()
        box.fill.fore_color.rgb = BOX
        box.line.color.rgb = ACCENT
        tf = box.text_frame
        tf.word_wrap = True
        for j, line in enumerate([head, detail]):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            run = p.add_run()
            run.text = line
            run.font.size = Pt(14 if j == 0 else 11)
            run.font.bold = j == 0
            run.font.color.rgb = INK if j == 0 else MUTED
        if i < len(steps) - 1:
            arrow = slide.shapes.add_shape(33, Inches(x + w + 0.03), Inches(y + 0.47),
                                           Inches(gap - 0.06), Inches(0.22))   # right arrow
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = ACCENT
            arrow.line.fill.background()


def picture(slide, path, x, y, w=None, h=None):
    """Add an image scaled to width ``w`` or height ``h`` (inches)."""
    return slide.shapes.add_picture(path, Inches(x), Inches(y),
                                    width=Inches(w) if w else None,
                                    height=Inches(h) if h else None)


def notes(slide, text):
    """Speaker notes."""
    slide.notes_slide.notes_text_frame.text = text


def slide_data(prs):
    """Slide 1: the English data and the twins."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Data and twins", "Public fact-check data, cleaned, plus copies that change exactly one thing")
    counts = {(r["dataset"], r["split"]): r for r in read_csv("reports/tables/tab_data_counts.csv")}
    rows = [["Dataset", "train", "val", "test", "snippets"]]
    for ds, name in [("multifc", "MultiFC"), ("averitec", "AVeriTeC")]:
        splits = [counts[(ds, sp)] for sp in ("train", "val", "test")]
        rows.append([name] + [n(r["claims"]) for r in splits]
                    + [n(sum(int(r["snippets"]) for r in splits))])
    table(s, 0.5, 1.6, 6.2, rows, [1.6, 1.1, 1.0, 1.0, 1.5])

    twins = {r["split"]: r for r in read_csv("reports/tables/tab_twins.csv")}
    t, tr = twins["test"], twins["train"]
    rows = [["Twin", "What changes", "Correct verdict", "test / train"],
            ["_P", "claim reworded, same names and numbers", "same", f"{n(t['_P'])} / {n(tr['_P'])}"],
            ["_B", "one place, person or year swapped in the evidence", "contradicted",
             f"{n(t['_B'])} / {n(tr['_B'])}"],
            ["_C", "evidence from an unrelated claim", "unverified", f"{n(t['_C'])} / {n(tr['_C'])}"]]
    table(s, 0.5, 3.0, 6.2, rows, [0.6, 3.0, 1.4, 1.2], size=12)
    text_box(s, 0.5, 5.25, 6.2, 1.6, [
        f"Sources: {n(t['sources'])} test + {n(tr['sources'])} train supported claims (seed 42).",
        "Twins keep the split of their source; automatic checks: 0 failures in test.",
        "Arabic _A / _M twins: Youssef. Image _V twins: Feba."], size=13, color=MUTED)
    picture(s, "reports/figures/fig_twin_counts.png", 7.0, 1.6, w=5.8)
    notes(s, "MultiFC and AVeriTeC give us real claims with web evidence. We removed snippets "
             "from fact-checking sites and snippets that just repeat the claim, and made our own "
             "80/10/10 split for MultiFC because its test labels were never released. A twin "
             "changes exactly one thing. A paraphrase should keep the verdict; an evidence swap "
             "should flip it to contradicted; unrelated evidence should give unverified. If a "
             "model still says 'supported' after the swap, it was not reading the evidence.")


def slide_demo(prs):
    """Slide 2: the demo pipeline and a real result card."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Demo: from a message to a cited verdict",
          "English, Arabic or mixed; every component real on one Colab T4")
    claims = {r["extractor"]: r for r in read_csv("reports/tables/tab_claims.csv")}
    f1 = float(claims["Qwen3-8B few-shot"]["all F1"])
    rec = {r["language"]: float(r["hybrid + reranker"]) for r in read_csv("reports/tables/tab_retrieval.csv")}
    library = json.loads(Path("results/R5/retrieval_eval.json").read_text(encoding="utf-8"))
    n_lib = library["n_snippets"]
    flow(s, 1.6, [("Message", "WhatsApp-style text"),
                  ("Claims", f"Qwen3-8B few-shot, F1 {f1:.2f}"),
                  ("Evidence", f"offline library, {n(n_lib)} snippets; recall@5 {rec['en']:.2f} EN / {rec['ar']:.2f} AR"),
                  ("NLI", "mDeBERTa-v3 per snippet"),
                  ("Verdict", "+ calibrated confidence"),
                  ("Cited note", "1–2 sentences, cites [1]–[3]")])
    picture(s, "reports/slides/demo_cards.png", 0.5, 3.0, h=4.2)
    rule = json.loads(Path("results/R6/examples_rule.json").read_text(encoding="utf-8"))
    head = json.loads(Path("results/R6/examples.json").read_text(encoding="utf-8"))

    def right(rep):
        return sum(all(c["label"] == ex["expected"] for c in ex["claims"]) and bool(ex["claims"])
                   for ex in rep["examples"])
    text_box(s, 6.3, 3.1, 6.5, 4.0, [
        "6 made-up UAE examples, labelled on screen: 2 supported, 2 contradicted, "
        "2 unverified; 2 Arabic, 1 mixed",
        "Arabic claims and notes right-to-left; confidence bar; citations link to the source",
        "No evidence pasted → search the offline library (BM25 + BGE-M3 → reranker → best 5)",
        f"Honest check: the evidence rule gets {right(rule)} of 6 examples right, "
        f"the trained verdict head {right(head)} of 6; the footer shows which one runs"],
        size=15)
    notes(s, "The demo takes a message, splits it into claims, finds evidence in our offline "
             "library, scores each snippet with NLI, and gives a verdict with a confidence and a "
             "short note. The numbers in the note are the snippets it used. The examples are made "
             "up and the app says so. Arabic is shown right to left. We also say openly which "
             "verdict step is running, because the trained head is weaker on these examples.")


def slide_repo(prs):
    """Slide 3: the repository and how to reproduce."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title(s, "Reproducible repository", "github.com/himaallu/Group-Project · one command per stage")
    tree = ["src/", "  data/  copies/  retrieval/  nli/", "  verdict/  explain/  claims/",
            "  translate/  vision/  eval/  demo/", "  pipeline.py   check(message)",
            "data/  features/  results/<task>/", "reports/{tables,figures}/",
            "docs/implementation.md", "notebooks/   tests/   run_all.sh"]
    box = s.shapes.add_shape(1, Inches(0.5), Inches(1.6), Inches(5.6), Inches(3.6))
    box.fill.solid()
    box.fill.fore_color.rgb = BOX
    box.line.fill.background()
    text_box(s, 0.7, 1.75, 5.3, 4.0, tree, size=15, font="Consolas")
    text_box(s, 6.5, 1.6, 6.3, 5.0, [
        "bash run_all.sh  → unit tests + every table and figure (CPU, minutes)",
        "STAGES=all  → data, reranker, twins, Arabic, NLI, verdict, notes, demo (GPU)",
        "Unit tests on 10-row fixtures for every module",
        "Fresh-clone test: run_all.sh rebuilds the tables and figures",
        "Fine-tuned models + offline library on the Hugging Face Hub "
        "(RathiAbeth/twincheck-reranker-v1, -nli-v1, -nli-v2, -index-v1)",
        "Every run in W&B project twincheck (seed 42; 43, 44 for repeats)",
        "Every task: results/<task>/README.md, notes/<task>.md, contribution log, PR"],
        size=15)
    notes(s, "Everything is in one GitHub repository. run_all.sh rebuilds the tables and "
             "figures from the result files in a few minutes, and the GPU stages rebuild the "
             "data, models and results from scratch. Large models live on the Hugging Face "
             "Hub. docs/implementation.md lists every model, setting and training time.")


def main() -> None:
    """Build the three slides."""
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)   # 16:9
    slide_data(prs)
    slide_demo(prs)
    slide_repo(prs)
    prs.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
