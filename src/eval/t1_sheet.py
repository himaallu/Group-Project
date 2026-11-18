"""Human-evaluation sheet for the cited notes (R8 for task T1, CLAUDE.md §9).

What it does
    Samples 60 notes from R5's scored notes, 40 English and 20 Arabic (seed 42),
    and writes the rating sheet. English notes are rated by Himasri AND Feba (so H6
    can compute Cohen's kappa); Arabic notes by Youssef. Each (note, rater) pair is
    one row, and each rater also gets their own file. The NLI faithfulness result is
    NOT on the sheet, so raters are not influenced by it.
    Raters fill three columns:
        verdict_correct       yes / no   (is the verdict right given the evidence?)
        sentences_supported   yes / partly / no   (is every sentence backed by the
                              snippet it cites?)
        readability           1-5
    and may add a comment.

Inputs
    --notes results/R5/notes_scored.jsonl (or results/R5/notes.jsonl)

Outputs
    results/T1/sheet.csv and results/T1/sheet_{himasri,feba,youssef}.csv
    (UTF-8 with BOM so Excel shows Arabic correctly). Filled sheets are saved by the
    raters as results/T1/human_eval.csv (same columns).

How to run (repo root, CPU)
    python -m src.eval.t1_sheet --n-en 4 --n-ar 2 --out-dir results/T1/test10   # quick check
    python -m src.eval.t1_sheet
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from src.eval.loaders import load_jsonl

RATERS = {"en": ("himasri", "feba"), "ar": ("youssef",)}
COLUMNS = ("note_id", "language", "rater", "claim", "verdict", "note",
           "snippet_1", "snippet_2", "snippet_3",
           "verdict_correct", "sentences_supported", "readability", "comment")
MAX_SNIPPET_CHARS = 400


def sample_notes(notes: list[dict], n_en: int, n_ar: int, seed: int = 42) -> list[dict]:
    """``n_en`` English and ``n_ar`` Arabic/mixed notes, sorted by id then shuffled."""
    rng = random.Random(seed)
    picked = []
    for langs, n in ((("en",), n_en), (("ar", "mixed"), n_ar)):
        pool = sorted((r for r in notes if r["language"] in langs), key=lambda r: r["id"])
        rng.shuffle(pool)
        if len(pool) < n:
            print(f"warning: only {len(pool)} notes for {langs}, wanted {n}")
        picked += pool[:n]
    return picked


def sheet_rows(notes: list[dict]) -> list[dict]:
    """One row per (note, rater), with empty rating columns."""
    rows = []
    for note in notes:
        lang = "en" if note["language"] == "en" else "ar"
        snippets = [s["text"][:MAX_SNIPPET_CHARS] for s in note["snippets"]][:3]
        snippets += [""] * (3 - len(snippets))
        for rater in RATERS[lang]:
            rows.append({"note_id": note["id"], "language": note["language"], "rater": rater,
                         "claim": note["claim"], "verdict": note["label"], "note": note["note"],
                         "snippet_1": snippets[0], "snippet_2": snippets[1],
                         "snippet_3": snippets[2], "verdict_correct": "",
                         "sentences_supported": "", "readability": "", "comment": ""})
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    """CSV with a BOM (Excel then reads Arabic correctly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI: write the full sheet and one sheet per rater."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--notes", default="results/R5/notes_scored.jsonl")
    ap.add_argument("--n-en", type=int, default=40)
    ap.add_argument("--n-ar", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/T1")
    args = ap.parse_args()

    rows = sheet_rows(sample_notes(load_jsonl(args.notes), args.n_en, args.n_ar, args.seed))
    out = Path(args.out_dir)
    write_csv(rows, out / "sheet.csv")
    for rater in sorted({r["rater"] for r in rows}):
        write_csv([r for r in rows if r["rater"] == rater], out / f"sheet_{rater}.csv")
    print(f"{len(rows)} rows -> {out}/sheet.csv (+ one file per rater)")


if __name__ == "__main__":
    main()
