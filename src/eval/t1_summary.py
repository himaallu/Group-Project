"""T1 human evaluation: combine the rating sheets, inter-rater agreement and averages (H6).

What it does
    Reads every filled ``results/T1/sheet_<rater>.csv`` (40 English notes rated by
    both Himasri and Feba, 20 Arabic notes by Youssef), writes them as one long table
    and reports:
      * agreement on the notes both English raters rated: Cohen's kappa for
        ``verdict_correct`` (yes/no) and ``sentences_supported`` (yes/partly/no),
        quadratic-weighted kappa for ``readability`` (1-5, ordinal), raw agreement;
      * averages per rater and language: share of verdicts judged correct, share of
        notes fully / partly / not supported, mean readability.
    Blank rows (a sheet not rated yet) are skipped and counted.

Inputs
    results/T1/sheet_*.csv (from src/eval/t1_sheet.py, filled by the raters)

Outputs
    results/T1/human_eval.csv          all ratings, one row per (note, rater)
    results/T1/human_eval_summary.json agreement + averages

How to run
    python -m src.eval.t1_summary
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

T1 = Path("results/T1")
FIELDS = ("verdict_correct", "sentences_supported", "readability")


def load_ratings() -> tuple[list[dict], dict[str, int]]:
    """All filled rating rows from the per-rater sheets, plus blank counts per sheet."""
    rows, blanks = [], {}
    for path in sorted(T1.glob("sheet_*.csv")):
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            if all((r.get(f) or "").strip() for f in FIELDS):
                rows.append({k: (v or "").strip() for k, v in r.items()})
            else:
                blanks[path.name] = blanks.get(path.name, 0) + 1
    return rows, blanks


def agreement(rows: list[dict], a: str, b: str) -> dict:
    """Kappa and raw agreement between raters ``a`` and ``b`` on the notes both rated."""
    by = {(r["note_id"], r["rater"]): r for r in rows}
    shared = sorted({n for n, r in by if r == a} & {n for n, r in by if r == b})
    out = {"raters": [a, b], "n_shared": len(shared)}
    if not shared:
        return out
    for f in FIELDS:
        x = [by[(n, a)][f] for n in shared]
        y = [by[(n, b)][f] for n in shared]
        weights = "quadratic" if f == "readability" else None
        kappa = cohen_kappa_score(x, y, weights=weights)
        out[f] = {"kappa": round(float(kappa), 3), "weighted": weights,
                  "agreement": round(sum(i == j for i, j in zip(x, y)) / len(x), 3)}
    return out


def averages(rows: list[dict]) -> dict:
    """Per rater and language: share correct, supported shares, mean readability."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(f"{r['rater']}/{r['language']}", []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        n = len(rs)
        out[key] = {"n": n,
                    "verdict_correct": round(sum(r["verdict_correct"] == "yes" for r in rs) / n, 3),
                    **{f"supported_{v}": round(sum(r["sentences_supported"] == v for r in rs) / n, 3)
                       for v in ("yes", "partly", "no")},
                    "readability_mean": round(sum(int(r["readability"]) for r in rs) / n, 2)}
    return out


def main() -> None:
    """Write the combined table and the summary JSON."""
    rows, blanks = load_ratings()
    with open(T1 / "human_eval.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"n_ratings": len(rows), "blank_rows": blanks,
               "agreement_en": agreement(rows, "himasri", "feba"), "averages": averages(rows)}
    (T1 / "human_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
