"""Turn every prediction file into metric tables (CSV + LaTeX).

What it does
    Finds all ``results/*/preds_*.jsonl``, joins each with the gold records,
    computes every metric with ``src.eval.metrics.compute_all`` and writes tables.
    Files named ``preds_<model>_s42.jsonl``, ``_s43``, ``_s44`` are treated as seeds
    of one model and reported as mean ± sd.

Inputs
    --results   folder to search (default ``results``); pattern ``*/preds_*.jsonl``
    --gold      one or more shared-format files with the gold labels (originals and
                twins), e.g. ``data/processed/en/multifc_test.jsonl
                data/copies/en/test.jsonl``
    --out       output folder (default ``reports/tables``)

Outputs (in --out)
    metrics_all.csv       one row per prediction file (task, model, seed, all metrics)
    metrics_by_model.csv  one row per (task, model): mean and sd over seeds
    main_results.tex      macro-F1, AUROC, ECE on original claims
    twin_results.tex      ESR, unrelated rate, paraphrase / cross-lingual consistency, ALR

How to run
    python -m src.eval.report --gold data/processed/en/multifc_test.jsonl data/copies/en/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.eval.loaders import join_gold, load_jsonl, load_preds
from src.eval.metrics import compute_all

SEED_PATTERN = re.compile(r"^(?P<model>.+)_s(?P<seed>\d+)$")

# (column, header) pairs for the two LaTeX tables. Values are rates, shown in %.
MAIN_COLUMNS = [("macro_f1", "Macro-F1"), ("auroc", "AUROC"), ("ece", "ECE")]
TWIN_COLUMNS = [("esr", "ESR"), ("unrelated_rate", "Unrelated"),
                ("paraphrase_consistency", "Para. cons."),
                ("crosslingual_consistency", "X-ling cons."), ("alr", "ALR")]
METRIC_COLUMNS = [c for c, _ in MAIN_COLUMNS + TWIN_COLUMNS] + ["crosslingual_A", "crosslingual_M"]


def parse_pred_path(path: Path) -> tuple[str, str, int | None]:
    """Get (task, model, seed) from ``results/<task>/preds_<model>[_s<seed>].jsonl``."""
    name = path.stem.removeprefix("preds_")
    match = SEED_PATTERN.match(name)
    if match:
        return path.parent.name, match["model"], int(match["seed"])
    return path.parent.name, name, None


def evaluate_file(path: Path, gold_records: list[dict]) -> dict:
    """Compute all metrics for one predictions file.

    Args:
        path: a ``preds_*.jsonl`` file.
        gold_records: gold records (originals and twins) to join with.

    Returns:
        Dict with ``task, model, seed, n_preds, n_missing_gold`` and every metric
        from ``compute_all``.
    """
    task, model, seed = parse_pred_path(path)
    preds = load_preds(path)
    df, missing = join_gold(preds, gold_records)
    if missing:
        print(f"  warning: {path}: {len(missing)} of {len(preds)} ids have no gold "
              f"record (e.g. {missing[:3]})")
    return {"task": task, "model": model, "seed": seed, "n_preds": len(preds),
            "n_missing_gold": len(missing), **compute_all(df)}


def aggregate_seeds(per_file: pd.DataFrame) -> pd.DataFrame:
    """Average metrics over seeds: one row per (task, model).

    Args:
        per_file: output rows of :func:`evaluate_file` as a DataFrame.

    Returns:
        DataFrame with ``task, model, n_seeds`` and ``<metric>_mean``,
        ``<metric>_sd`` (sample sd; 0 for one seed, NaN if the metric is missing).
    """
    rows = []
    for (task, model), group in per_file.groupby(["task", "model"], sort=True):
        row = {"task": task, "model": model, "n_seeds": len(group),
               "n_orig": int(group["n_orig"].iloc[0])}
        for col in METRIC_COLUMNS:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}_mean"] = values.mean() if len(values) else float("nan")
            row[f"{col}_sd"] = values.std(ddof=1) if len(values) > 1 else (
                0.0 if len(values) else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def _latex_escape(text: str) -> str:
    """Escape the LaTeX special characters that appear in model and task names."""
    for char in ("\\", "_", "&", "%", "#", "$"):
        text = text.replace(char, "\\" + char)
    return text


def format_cell(mean: float, sd: float, n_seeds: int) -> str:
    """Format a rate as a percentage, with ± sd when there is more than one seed."""
    if pd.isna(mean):
        return "--"
    if n_seeds > 1:
        return f"{100 * mean:.1f} $\\pm$ {100 * sd:.1f}"
    return f"{100 * mean:.1f}"


def to_latex(by_model: pd.DataFrame, columns: list[tuple[str, str]], caption: str,
             label: str) -> str:
    """Build a booktabs LaTeX table from the aggregated metrics.

    Args:
        by_model: output of :func:`aggregate_seeds`.
        columns: (metric column, header) pairs.
        caption: table caption.
        label: LaTeX label, e.g. ``tab:main``.

    Returns:
        LaTeX source of a ``table`` environment.
    """
    header = " & ".join(["Task", "Model"] + [h for _, h in columns])
    lines = [
        "\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{ll{'r' * len(columns)}}}", "\\toprule",
        header + " \\\\", "\\midrule",
    ]
    for _, row in by_model.iterrows():
        cells = [_latex_escape(row["task"]), _latex_escape(row["model"])]
        cells += [format_cell(row[f"{c}_mean"], row[f"{c}_sd"], row["n_seeds"])
                  for c, _ in columns]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def is_verdict_preds(path: Path) -> bool:
    """True if the file's first record looks like a verdict prediction (§5 format).

    Other tasks also write ``preds_*.jsonl`` (e.g. Youssef's claim-extraction
    ``results/Y3/preds_qwen.jsonl`` with ``pred_claims``); those are skipped.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                return "pred_label" in rec and "probs" in rec
    return False


def build_report(results_dir: Path, gold_paths: list[Path], out_dir: Path) -> pd.DataFrame:
    """Evaluate every predictions file and write all tables.

    Args:
        results_dir: folder containing ``<task>/preds_*.jsonl``.
        gold_paths: shared-format gold files.
        out_dir: where the CSV and LaTeX files go (created if needed).

    Returns:
        The per-file metrics table (also written to ``metrics_all.csv``).

    Raises:
        FileNotFoundError: if a gold file is missing or no predictions file is found.
    """
    gold_records = []
    for gold_path in gold_paths:
        if not gold_path.exists():
            raise FileNotFoundError(f"gold file not found: {gold_path}")
        gold_records += load_jsonl(gold_path)

    pred_paths = sorted(results_dir.glob("*/preds_*.jsonl"))
    skipped = [p for p in pred_paths if not is_verdict_preds(p)]
    pred_paths = [p for p in pred_paths if p not in skipped]
    for p in skipped:
        print(f"skipping {p}: not a verdict predictions file")
    if not pred_paths:
        raise FileNotFoundError(f"no {results_dir}/*/preds_*.jsonl files found")

    print(f"{len(gold_records)} gold records, {len(pred_paths)} prediction files")
    per_file = pd.DataFrame([evaluate_file(p, gold_records) for p in pred_paths])
    by_model = aggregate_seeds(per_file)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_file.to_csv(out_dir / "metrics_all.csv", index=False)
    by_model.to_csv(out_dir / "metrics_by_model.csv", index=False)
    (out_dir / "main_results.tex").write_text(to_latex(
        by_model, MAIN_COLUMNS,
        "Results on original claims (\\%, mean $\\pm$ sd over seeds).", "tab:main"))
    (out_dir / "twin_results.tex").write_text(to_latex(
        by_model, TWIN_COLUMNS,
        "Twin metrics (\\%). ESR, unrelated and consistency: higher is better; "
        "ALR: lower is better.", "tab:twins"))
    print(f"wrote metrics_all.csv, metrics_by_model.csv, main_results.tex, "
          f"twin_results.tex to {out_dir}")
    return per_file


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build metric tables from prediction files.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--gold", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/tables"))
    args = parser.parse_args()
    build_report(args.results, args.gold, args.out)


if __name__ == "__main__":
    main()
