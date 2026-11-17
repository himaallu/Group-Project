"""Make the twin hand-check sheets for Himasri and Feba (H4 / F4, CLAUDE.md §4).

What it does
    Samples test twins and writes one CSV sheet per rater. Each rater gets
    ``n_per_rater`` twins (default 75); ``overlap`` of them (default 30) appear on
    both sheets so ``handcheck_kappa.py`` can compute Cohen's kappa. In total
    2 * 75 - 30 = 120 different twins are checked.
    Sampling is balanced across twin types (equal share per type; if a type has too
    few twins the rest goes to the other types) and fixed with seed 42. The order of
    shared and own rows is shuffled so a rater cannot tell which rows overlap.

Inputs
    --twins      twin files in the shared format (e.g. data/copies/en/test.jsonl)
    --originals  the source records the twins were made from
                 (e.g. data/processed/en/multifc_test.jsonl)
    Only ``split == "test"`` twins are used; anything else is refused.

Outputs
    results/H4/handcheck_himasri.csv and results/H4/handcheck_feba.csv (UTF-8 with
    BOM so Excel shows Arabic correctly). Columns:
        id, twin_type, orig_claim, twin_claim, orig_evidence, twin_evidence,
        orig_label, expected_label, twin_label, valid, note
    Raters fill ``valid`` with y / n (is this a correct twin that changes exactly
    one thing, with ``expected_label`` the right verdict?) and an optional ``note``.

How to run
    python -m src.copies.handcheck_sheet --twins data/copies/en/test.jsonl \
        --originals data/processed/en/multifc_test.jsonl --out-dir results/H4
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from src.eval.loaders import load_jsonl, split_twin_id, twin_type

RATERS = ("himasri", "feba")
COLUMNS = ("id", "twin_type", "orig_claim", "twin_claim", "orig_evidence",
           "twin_evidence", "orig_label", "expected_label", "twin_label", "valid", "note")
MAX_SNIPPETS = 3        # snippets shown per side, to keep the sheet readable
MAX_SNIPPET_CHARS = 300


def expected_label(kind: str, orig_label: str) -> str:
    """The verdict a correct twin should get (CLAUDE.md §2 table).

    ``_B`` -> contradicted, ``_C`` -> unverified, every other twin keeps the
    original's label.
    """
    if kind == "B":
        return "contradicted"
    if kind == "C":
        return "unverified"
    return orig_label


def evidence_diff(orig_ev: list[dict], twin_ev: list[dict]) -> tuple[str, str]:
    """Show only the snippets that differ between original and twin.

    Args:
        orig_ev, twin_ev: ``evidence`` lists from the two records.

    Returns:
        (orig text, twin text). Snippets are joined with `` || `` and cut to
        MAX_SNIPPET_CHARS. If nothing differs (e.g. ``_P`` twins), both sides say
        ``(same evidence)`` followed by the first snippet for context.
    """
    def short(s: str) -> str:
        return s if len(s) <= MAX_SNIPPET_CHARS else s[:MAX_SNIPPET_CHARS] + "..."

    o = [e.get("text", "") for e in orig_ev]
    t = [e.get("text", "") for e in twin_ev]
    o_only = [s for s in o if s not in set(t)][:MAX_SNIPPETS]
    t_only = [s for s in t if s not in set(o)][:MAX_SNIPPETS]
    if not o_only and not t_only:
        first = short(o[0]) if o else ""
        return f"(same evidence) {first}", "(same evidence)"
    return " || ".join(short(s) for s in o_only), " || ".join(short(s) for s in t_only)


def make_row(twin: dict, orig: dict) -> dict:
    """One sheet row for a twin and its original (``valid`` and ``note`` empty)."""
    kind = twin_type(twin["id"])
    o_text, t_text = evidence_diff(orig.get("evidence", []), twin.get("evidence", []))
    return {
        "id": twin["id"], "twin_type": kind,
        "orig_claim": orig["claim"], "twin_claim": twin["claim"],
        "orig_evidence": o_text, "twin_evidence": t_text,
        "orig_label": orig["label"], "expected_label": expected_label(kind, orig["label"]),
        "twin_label": twin["label"], "valid": "", "note": "",
    }


def balanced_sample(twins: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Pick ``n`` twins with an equal share per twin type, in round-robin order.

    Round-robin order (P, B, C, P, B, C, ...) means every prefix of the result is
    also balanced, so the first ``overlap`` rows cover all types.

    Raises:
        ValueError: if there are fewer than ``n`` twins.
    """
    if len(twins) < n:
        raise ValueError(f"need {n} twins, only {len(twins)} available")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for tw in twins:
        by_type[twin_type(tw["id"])].append(tw)
    for kind in sorted(by_type):
        rng.shuffle(by_type[kind])
    # Take one twin per type in turn until n are picked; empty types drop out,
    # which gives their share to the other types.
    picked, pos = [], 0
    kinds = sorted(by_type)
    while len(picked) < n:
        for kind in kinds:
            if pos < len(by_type[kind]) and len(picked) < n:
                picked.append(by_type[kind][pos])
        pos += 1
    return picked


def build_sheets(twins: list[dict], originals: list[dict], n_per_rater: int = 75,
                 overlap: int = 30, seed: int = 42) -> dict[str, list[dict]]:
    """Build both raters' sheets.

    Args:
        twins: twin records (test split only).
        originals: source records; every twin's base id must be among them.
        n_per_rater: rows per sheet.
        overlap: rows shared by both sheets (for kappa).
        seed: random seed (42).

    Returns:
        {"himasri": rows, "feba": rows}, each a list of row dicts (COLUMNS).

    Raises:
        ValueError: on non-test twins, missing originals, or bad sizes.
    """
    if not 0 <= overlap <= n_per_rater:
        raise ValueError("overlap must be between 0 and n_per_rater")
    bad = sorted({tw.get("split") for tw in twins} - {"test"})
    if bad:
        raise ValueError(f"hand-check uses test twins only; got splits {bad}")
    by_id = {o["id"]: o for o in originals}
    missing = [tw["id"] for tw in twins if split_twin_id(tw["id"])[0] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} twins have no original, e.g. {missing[:3]}")

    rng = random.Random(seed)
    total = 2 * n_per_rater - overlap
    picked = balanced_sample(twins, total, rng)
    shared = picked[:overlap]
    own = n_per_rater - overlap
    sheets = {RATERS[0]: shared + picked[overlap:overlap + own],
              RATERS[1]: shared + picked[overlap + own:]}
    result = {}
    for rater, rows in sheets.items():
        rows = [make_row(tw, by_id[split_twin_id(tw["id"])[0]]) for tw in rows]
        rng.shuffle(rows)  # hide which rows are shared
        result[rater] = rows
    return result


def write_sheet(rows: list[dict], path: str | Path) -> None:
    """Write one sheet as CSV (UTF-8 with BOM for Excel)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI: read twins + originals, write one sheet per rater."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--twins", nargs="+", required=True)
    ap.add_argument("--originals", nargs="+", required=True)
    ap.add_argument("--out-dir", default="results/H4")
    ap.add_argument("--n-per-rater", type=int, default=75)
    ap.add_argument("--overlap", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    twins = [r for p in args.twins for r in load_jsonl(p)]
    originals = [r for p in args.originals for r in load_jsonl(p)]
    sheets = build_sheets(twins, originals, args.n_per_rater, args.overlap, args.seed)
    for rater, rows in sheets.items():
        out = Path(args.out_dir) / f"handcheck_{rater}.csv"
        write_sheet(rows, out)
        counts = defaultdict(int)
        for r in rows:
            counts[r["twin_type"]] += 1
        print(f"wrote {len(rows)} rows to {out}  by type {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
