"""Every report table (CSV + LaTeX) and figure (PNG, 300 dpi) from the files in results/ (R8).

What it does
    Reads the small result files each task pushes and writes one table and/or figure
    for each. A source that is not there yet is skipped with a message, so the script
    can be rerun as results arrive. Every number comes from a file in results/
    (CLAUDE.md §10); nothing is typed in by hand.

        source file(s)                          -> table / figure
        results/R1/counts.csv                   -> tab_data_counts
        results/R2/reranker_eval.json           -> tab_reranker
        results/R3/make_twins_report.json       -> tab_twins, fig_twin_counts
        results/R4/gold_eval*.json              -> tab_nli, fig_nli_gold
        results/R5/{explain,retrieval}_eval.json-> tab_explain, tab_retrieval
        reports/tables/metrics_by_model.csv     -> fig_twin_metrics (from src/eval/report.py)
        results/*/preds_*_s42.jsonl (+ gold)    -> fig_reliability (calibration curves)
        results/Y5/arabic_E7.csv                -> tab_arabic_E7
        results/F4/audit_E0.csv                 -> tab_audit_E0

    Colours: the first categorical slots of the shared chart palette, in fixed order
    (blue, orange, aqua, yellow), so a model keeps its colour in every figure.

Inputs
    results/ and reports/tables/metrics_by_model.csv (run src.eval.report first).

Outputs
    reports/tables/<name>.{csv,tex} and reports/figures/<name>.png

How to run (repo root, CPU)
    python -m src.eval.figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# --------------------------------------------------------------------------
# Writing tables
# --------------------------------------------------------------------------

def latex_escape(value) -> str:
    """Escape LaTeX special characters in a cell."""
    text = str(value)
    for char, repl in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                       ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        text = text.replace(char, repl)
    return text


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """A booktabs LaTeX table (no extra packages beyond booktabs)."""
    cols = "l" + "r" * (len(df.columns) - 1)
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{latex_escape(caption)}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(latex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    for row in df.itertuples(index=False):
        lines.append(" & ".join(latex_escape(fmt(v)) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def fmt(value) -> str:
    """Numbers to 3 decimals (integers as they are), missing values as a dash."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_table(df: pd.DataFrame, name: str, caption: str, out: Path) -> None:
    """Write ``df`` as reports/tables/<name>.csv and .tex."""
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.tex").write_text(to_latex(df, caption, f"tab:{name}"), encoding="utf-8")
    print(f"  table  {name} ({len(df)} rows)")


# --------------------------------------------------------------------------
# Building tables from result files
# --------------------------------------------------------------------------

def read_json(path: Path) -> dict | None:
    """JSON content, or None if the file is not there yet."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def twins_table(report: dict) -> pd.DataFrame:
    """R3 twin counts per split."""
    rows = []
    for split in ("test", "train"):
        r = report.get(split)
        if r:
            rows.append({"split": split, "sources": r["n_sources"], "_P": r["P_made"],
                         "_B": r["B_made"], "_C": r["C_made"], "twins": r["n_twins"]})
    return pd.DataFrame(rows)


def reranker_table(ev: dict) -> pd.DataFrame:
    """R2 reranker before / after fine-tuning on the val pairs."""
    return pd.DataFrame([{"model": name, "accuracy": ev[key]["accuracy"], "AUROC": ev[key]["auroc"],
                          "mean rel. (own)": ev[key]["mean_relevance_pos"],
                          "mean rel. (random)": ev[key]["mean_relevance_neg"]}
                         for name, key in (("zero-shot", "val_zero_shot"),
                                           ("fine-tuned", "val_fine_tuned"))])


def nli_table(evals: dict[str, dict]) -> pd.DataFrame:
    """R4 macro-F1 for each model, zero-shot and fine-tuned, per evaluation set."""
    rows = []
    for model, ev in evals.items():
        for when in ("zero_shot", "fine_tuned"):
            row = {"model": model, "setting": when.replace("_", "-")}
            for key, res in ev[when].items():
                row[f"{key} macro-F1"] = res["macro_f1"]
            rows.append(row)
    return pd.DataFrame(rows)


def explain_table(ev: dict) -> pd.DataFrame:
    """R5 faithfulness and citation accuracy per language."""
    return pd.DataFrame([{"language": lang, "notes": v["n_notes"], "sentences": v["n_sentences"],
                          "faithfulness": v["faithfulness"],
                          "citation accuracy": v["citation_accuracy"]}
                         for lang, v in ev.items() if isinstance(v, dict) and "n_notes" in v])


def retrieval_table(ev: dict) -> pd.DataFrame:
    """R5 recall@5 per language and search method."""
    return pd.DataFrame([{"language": lang, "claims": ev[lang]["n_claims"],
                          "BM25": ev[lang]["recall@5_bm25"], "dense": ev[lang]["recall@5_dense"],
                          "hybrid + reranker": ev[lang]["recall@5_hybrid"]}
                         for lang in ("en", "ar") if lang in ev])


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def style_axes(ax) -> None:
    """Recessive grid and axes, text in ink colours (never series colours)."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def grouped_bars(groups: list[str], series: dict[str, list[float]], ylabel: str, title: str,
                 path: Path, ylim: tuple[float, float] | None = (0, 1)) -> None:
    """Grouped bar chart: one colour per series in fixed order, legend always shown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(series)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(4.0, 1.1 * len(groups) * max(n, 1) * 0.6 + 2), 3.2),
                           facecolor=SURFACE)
    for i, (name, values) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        vals = [0 if v is None or pd.isna(v) else v for v in values]
        # edgecolor = surface: a thin gap between neighbouring bars
        ax.bar(xs, vals, width * 0.92, color=SERIES[i % len(SERIES)], label=name,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    style_axes(ax)
    if n >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, ncol=min(n, 4),
                  loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"  figure {path.name}")


def reliability_points(df: pd.DataFrame, n_bins: int = 15) -> tuple[list, list]:
    """(mean confidence, accuracy) per non-empty bin for originals in a joined table."""
    from src.eval.metrics import LABELS, originals

    orig = originals(df)
    probs = orig[[f"p_{lab}" for lab in LABELS]].to_numpy()
    conf = probs.max(axis=1)
    correct = (orig["pred"] == orig["gold"]).to_numpy()
    xs, ys = [], []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if mask.any():
            xs.append(float(conf[mask].mean()))
            ys.append(float(correct[mask].mean()))
    return xs, ys


def reliability_figure(curves: dict[str, tuple[list, list]], path: Path) -> None:
    """Calibration curves (one line per model) against the diagonal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.8), facecolor=SURFACE)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect")
    for i, (name, (xs, ys)) in enumerate(curves.items()):
        ax.plot(xs, ys, color=SERIES[i % len(SERIES)], linewidth=2, marker="o", markersize=4,
                label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence", color=TEXT)
    ax.set_ylabel("accuracy")
    ax.set_title("Calibration on original test claims (seed 42)", fontsize=10, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"  figure {path.name}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_all(results: Path, tables: Path, figures: Path, gold_paths: list[str]) -> None:
    """Make every table and figure whose source exists (see the module docstring)."""
    def skip(what: str) -> None:
        print(f"  skip   {what} (source not in results/ yet)")

    counts = results / "R1" / "counts.csv"
    if counts.exists():
        write_table(pd.read_csv(counts), "tab_data_counts",
                    "English data after cleaning: claims per label and split, snippets.", tables)
    else:
        skip("tab_data_counts")

    ev = read_json(results / "R2" / "reranker_eval.json")
    if ev:
        write_table(reranker_table(ev), "tab_reranker",
                    "Reranker on 2,000 validation pairs before and after fine-tuning.", tables)
    else:
        skip("tab_reranker")

    twins = read_json(results / "R3" / "make_twins_report.json")
    if twins:
        df = twins_table(twins)
        write_table(df, "tab_twins", "English twins per split.", tables)
        # matplotlib hides legend labels that start with "_", so name them in words
        names = {"_P": "paraphrase (P)", "_B": "evidence swap (B)", "_C": "unrelated (C)"}
        grouped_bars(list(df["split"]), {names[t]: list(df[t]) for t in names},
                     "twins", "English twins per split and type", figures / "fig_twin_counts.png",
                     ylim=None)
    else:
        skip("tab_twins / fig_twin_counts")

    nli = {name: e for name, e in (("mDeBERTa-v3", read_json(results / "R4" / "gold_eval.json")),
                                   ("XLM-R-large", read_json(results / "R4" / "gold_eval_xlmr.json")))
           if e}
    if nli:
        df = nli_table(nli)
        write_table(df, "tab_nli", "NLI macro-F1, zero-shot vs fine-tuned (gold = hand-labelled pairs).",
                    tables)
        gold_cols = [c for c in df.columns if c.startswith("gold_")]
        series = {f"{r['model']} ({r['setting']})": [r[c] for c in gold_cols]
                  for _, r in df.iterrows()}
        labels = [c.replace(" macro-F1", "").replace("_", " ") for c in gold_cols]  # "gold en"
        grouped_bars(labels, series, "macro-F1",
                     "NLI on the gold pairs", figures / "fig_nli_gold.png")
    else:
        skip("tab_nli / fig_nli_gold")

    expl = read_json(results / "R5" / "explain_eval.json")
    if expl:
        write_table(explain_table(expl), "tab_explain",
                    "Cited notes: faithfulness and citation accuracy (fine-tuned NLI judge).", tables)
    else:
        skip("tab_explain")
    retr = read_json(results / "R5" / "retrieval_eval.json")
    if retr:
        df = retrieval_table(retr)
        write_table(df, "tab_retrieval", "Offline evidence search: recall@5.", tables)
        grouped_bars(list(df["language"]), {m: list(df[m]) for m in ("BM25", "dense", "hybrid + reranker")},
                     "recall@5", "Offline evidence search", figures / "fig_retrieval.png")
    else:
        skip("tab_retrieval")

    by_model = tables / "metrics_by_model.csv"
    if by_model.exists():
        df = pd.read_csv(by_model)
        metrics = [("esr_mean", "ESR"), ("unrelated_rate_mean", "Unrelated"),
                   ("paraphrase_consistency_mean", "Paraphrase"), ("crosslingual_A_mean", "Cross-ling.")]
        metrics = [(c, n) for c, n in metrics if c in df.columns and df[c].notna().any()]
        series = {f"{r['task']}/{r['model']}": [r[c] for c, _ in metrics] for _, r in df.iterrows()}
        grouped_bars([n for _, n in metrics], dict(list(series.items())[:6]), "rate",
                     "Twin metrics per model", figures / "fig_twin_metrics.png")
    else:
        skip("fig_twin_metrics (run src.eval.report first)")

    pred_files = sorted(results.glob("*/preds_*_s42.jsonl"))
    if pred_files and all(Path(p).exists() for p in gold_paths):
        from src.eval.loaders import join_gold, load_jsonl, load_preds
        gold = [r for p in gold_paths for r in load_jsonl(p)]
        curves = {}
        for path in pred_files[:6]:
            df, _ = join_gold(load_preds(path), gold)
            name = f"{path.parent.name}/{path.stem.removeprefix('preds_').removesuffix('_s42')}"
            curves[name] = reliability_points(df)
        reliability_figure(curves, figures / "fig_reliability.png")
    else:
        skip("fig_reliability")

    for src, name, caption in ((results / "Y5" / "arabic_E7.csv", "tab_arabic_E7",
                                "Arabic and mixed-language results (E7)."),
                               (results / "F4" / "audit_E0.csv", "tab_audit_E0",
                                "Audit of baselines with the English twins (E0).")):
        if src.exists():
            write_table(pd.read_csv(src), name, caption, tables)
        else:
            skip(name)


def main() -> None:
    """CLI: build everything."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--figures", default="reports/figures")
    ap.add_argument("--gold", nargs="+", default=["data/processed/en/multifc_test.jsonl",
                                                   "data/copies/en/test.jsonl"])
    args = ap.parse_args()
    build_all(Path(args.results), Path(args.tables), Path(args.figures), args.gold)


if __name__ == "__main__":
    main()
