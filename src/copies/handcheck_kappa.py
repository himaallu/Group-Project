"""Cohen's kappa and valid-twin rates from the two filled hand-check sheets (H4).

What it does
    Reads Himasri's and Feba's filled sheets (made by ``handcheck_sheet.py``),
    computes Cohen's kappa on the ``valid`` column of the rows both rated (target
    >= 0.7, CLAUDE.md §4), and the share of valid twins per twin type. A shared row
    counts as valid only if both raters said y.

Inputs
    Two CSV sheets with at least the columns ``id, twin_type, valid``; ``valid`` is
    y / n (also accepted: yes, no, 1, 0, any case).

Outputs
    results/H4/handcheck_summary.json:
        {"kappa": 0.78, "n_overlap": 30, "agreement": 0.93, "disagreements": [ids],
         "valid_rate": {"all": 0.91, "B": ..., "C": ..., "P": ...}, "n_rated": 120}
    The same numbers are printed.

How to run
    python -m src.copies.handcheck_kappa results/H4/handcheck_himasri.csv \
        results/H4/handcheck_feba.csv --out results/H4/handcheck_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.eval.metrics import cohen_kappa

YES = {"y", "yes", "1", "true"}
NO = {"n", "no", "0", "false"}


def normalise_valid(value: str) -> str | None:
    """Map a ``valid`` cell to "y" / "n", or None if it is blank.

    Raises:
        ValueError: for anything that is not a recognised yes/no answer.
    """
    v = (value or "").strip().lower()
    if not v:
        return None
    if v in YES:
        return "y"
    if v in NO:
        return "n"
    raise ValueError(f"valid must be y or n, got {value!r}")


def read_sheet(path: str | Path, allow_partial: bool = False) -> dict[str, dict]:
    """Read a filled sheet into {id: {"twin_type", "valid"}}.

    Args:
        path: CSV file.
        allow_partial: if False, any blank ``valid`` cell is an error.

    Raises:
        ValueError: on blanks (unless allowed), bad values, or duplicate ids.
    """
    rows: dict[str, dict] = {}
    blanks = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            try:
                valid = normalise_valid(row.get("valid", ""))
            except ValueError as err:
                raise ValueError(f"{path} line {line_no}: {err}") from None
            if valid is None:  # not rated yet
                blanks.append(row["id"])
                continue
            if row["id"] in rows:
                raise ValueError(f"{path}: duplicate id {row['id']}")
            rows[row["id"]] = {"twin_type": row["twin_type"], "valid": valid}
    if blanks and not allow_partial:
        raise ValueError(f"{path}: {len(blanks)} rows not rated yet, e.g. {blanks[:3]}")
    return rows


def summarise(sheet_a: dict[str, dict], sheet_b: dict[str, dict]) -> dict:
    """Kappa on shared rows + valid rate per twin type over all rated twins.

    Args:
        sheet_a, sheet_b: outputs of ``read_sheet``.

    Returns:
        Summary dict (see module docstring).
    """
    shared = sorted(set(sheet_a) & set(sheet_b))
    a = [sheet_a[i]["valid"] for i in shared]
    b = [sheet_b[i]["valid"] for i in shared]

    # One verdict per twin: shared rows are valid only if both raters said y.
    combined: dict[str, tuple[str, bool]] = {}
    for sheet in (sheet_a, sheet_b):
        for i, r in sheet.items():
            ok = r["valid"] == "y"
            prev = combined.get(i)
            combined[i] = (r["twin_type"], ok if prev is None else prev[1] and ok)

    by_type: dict[str, list[bool]] = defaultdict(list)
    for kind, ok in combined.values():
        by_type[kind].append(ok)
    all_ok = [ok for _, ok in combined.values()]
    valid_rate = {"all": sum(all_ok) / len(all_ok) if all_ok else None}
    valid_rate.update({k: sum(v) / len(v) for k, v in sorted(by_type.items())})

    return {
        "kappa": cohen_kappa(a, b),
        "n_overlap": len(shared),
        "agreement": sum(x == y for x, y in zip(a, b)) / len(shared) if shared else None,
        "disagreements": [i for i, x, y in zip(shared, a, b) if x != y],
        "valid_rate": valid_rate,
        "n_rated": len(combined),
    }


def main() -> None:
    """CLI: two sheets in, summary JSON out."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sheet_a")
    ap.add_argument("sheet_b")
    ap.add_argument("--out", default="results/H4/handcheck_summary.json")
    ap.add_argument("--allow-partial", action="store_true",
                    help="skip rows whose 'valid' cell is still blank")
    args = ap.parse_args()
    summary = summarise(read_sheet(args.sheet_a, args.allow_partial),
                        read_sheet(args.sheet_b, args.allow_partial))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["kappa"] is not None and summary["kappa"] < 0.7:
        print("WARNING: kappa below 0.7 target: discuss the disagreements and re-check.")


if __name__ == "__main__":
    main()
