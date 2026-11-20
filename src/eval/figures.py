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
        results/R4/{v1_,}gold_eval*.json        -> tab_nli, fig_nli_gold (v1, v2, XLM-R)
        results/R5/{explain,retrieval}_eval.json-> tab_explain, tab_retrieval, fig_retrieval
        results/H5/summary.csv                  -> tab_main, tab_averitec, fig_main
        results/H4/bootstrap_twin_vs_notwin.json-> tab_bootstrap
        results/H4/preds_*_s42.jsonl (+ gold)   -> fig_reliability (calibration curves)
        results/Y3/f1_{split,qwen}.json         -> tab_claims
        results/Y5/arabic_E7.csv                -> tab_arabic_E7
        results/F4/audit_E0.csv                 -> tab_audit_E0, tab_audit_averitec
        results/F3/F3_FINAL_SUMMARY.csv + F5/cnn_vit.csv -> tab_detectors, tab_robustness
        results/F3/F3_corrected_pair_ALR_summary.csv     -> tab_leakage
        results/F5/onnx_latency.csv             -> tab_onnx

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
    return text.replace("±", r"$\pm$").replace(">=", r"$\geq$")


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


def nli_table(evals: list[tuple[str, str, dict]]) -> pd.DataFrame:
    """R4 gold macro-F1: zero-shot once per base model, then each fine-tuned version.

    ``evals`` holds (base model, version, gold_eval dict). Only the hand-labelled gold
    sets are shown: the weak validation sets differ between versions.
    """
    rows, seen = [], set()
    for model, version, ev in evals:
        for when, setting in (("zero_shot", "zero-shot"), ("fine_tuned", f"fine-tuned {version}")):
            if when == "zero_shot" and model in seen:
                continue
            seen.add(model)
            row = {"model": model, "setting": setting}
            for key in ("gold_en", "gold_ar"):
                if key in ev[when]:
                    row[f"{key} macro-F1"] = ev[when][key]["macro_f1"]
            rows.append(row)
    return pd.DataFrame(rows)


# H5 summary.csv model names -> report names (fixed order = fixed colours)
MAIN_MODELS = {"notwin": "no twin loss", "twin": "twin loss",
               "twin_norel": "twin loss, no reranker", "twin_noent": "twin loss, no entity match"}
MAIN_METRICS = {"macro_f1": "macro-F1", "esr": "ESR", "unrelated_rate": "unrelated",
                "paraphrase_consistency": "paraphrase cons.", "ece": "ECE"}


def pct(value: float) -> str:
    """A rate as a percentage with one decimal, rounded half up (0.3625 -> '36.3')."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(str(round(100 * value, 6))).quantize(Decimal("0.1"), ROUND_HALF_UP))


def pm(mean, sd) -> str:
    """'mean ± sd' in %, one decimal; a dash if missing."""
    if mean is None or pd.isna(mean):
        return "–"
    return f"{pct(mean)} ± {pct(sd)}"


def main_table(summary: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Verdict-head variants (H5) as 'mean ± sd' over seeds, in %.

    ECE is shown before and after temperature scaling; ablations have no calibrated
    run, so their "ECE (cal.)" is a dash. Twin metrics are dropped for datasets
    without twins (AVeriTeC).
    """
    by = summary.set_index("model")
    rows = []
    for key, name in MAIN_MODELS.items():
        if prefix + key not in by.index:
            continue
        r, row = by.loc[prefix + key], {"model": name}
        for m, label in MAIN_METRICS.items():
            row[label] = pm(r.get(f"{m}_mean"), r.get(f"{m}_std"))
        cal = by.loc[prefix + key + "_cal"] if prefix + key + "_cal" in by.index else {}
        row["ECE (cal.)"] = pm(cal.get("ece_mean") if len(cal) else None,
                               cal.get("ece_std") if len(cal) else None)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[[c for c in df.columns if c == "model" or (df[c] != "–").any()]]


def bootstrap_table(boot: dict) -> pd.DataFrame:
    """Twin vs no-twin paired bootstrap per seed: difference, 95% CI and p, in points."""
    rows = []
    for pair, metrics in boot.items():
        seed = pair.rsplit("_s", 1)[-1]          # "preds_twin_s42 vs preds_notwin_s42" -> "42"
        for m, label in MAIN_METRICS.items():
            if m in metrics:
                b = metrics[m]
                rows.append({"seed": seed, "metric": label, "diff": round(100 * b["diff"], 1),
                             "95% CI": f"[{100 * b['ci_low']:.1f}, {100 * b['ci_high']:.1f}]",
                             "p": b["p_value"]})
    return pd.DataFrame(rows)


def claims_table(f1: dict[str, dict]) -> pd.DataFrame:
    """Y3 claim-extraction F1 per language for each extractor."""
    return pd.DataFrame([{"extractor": name, **{f"{lang} F1": ev[lang]["f1"]
                                                for lang in ("en", "ar", "mixed", "all") if lang in ev}}
                         for name, ev in f1.items()])


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


# F4 audit_E0.csv model names -> report names (fixed order)
BASELINES = {"claim_only": "claim-only", "similarity": "similarity + LightGBM",
             "zeroshot_nli": "zero-shot NLI", "llm_judge": "LLM judge (Qwen3-8B)"}
AUDIT_METRICS = {"macro_f1": "macro-F1", "auroc": "AUROC", "esr": "ESR",
                 "unrelated_rate": "unrelated", "paraphrase_consistency": "paraphrase cons.",
                 "ece": "ECE"}


def audit_table(audit: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """F4 baselines on one dataset, in % ('mean ± sd' when there are several seeds).

    AVeriTeC rows are named "avt_<model>" and have no twins, so their twin columns
    are dropped.
    """
    rows = []
    for key, name in BASELINES.items():
        hit = audit[(audit["dataset"] == dataset) & audit["model"].isin([key, f"avt_{key}"])]
        if hit.empty:
            continue
        r = hit.iloc[0]
        row = {"baseline": name, "seeds": int(r["n_seeds"])}
        for m, label in AUDIT_METRICS.items():
            sd = r.get(f"{m}_sd")
            row[label] = pm(r.get(m), sd) if pd.notna(sd) else (
                "–" if pd.isna(r.get(m)) else pct(r.get(m)))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[[c for c in df.columns if c in ("baseline", "seeds") or (df[c] != "–").any()]]


DETECTORS = {"resnet50": "ResNet-50 (last 4 blocks)", "vit_b16": "ViT-B/16 (last 4 blocks)"}


def detector_table(probe: pd.DataFrame, cnn_vit: pd.DataFrame) -> pd.DataFrame:
    """F3 CLIP probes and F5 CNN / ViT on the unseen-generator test (real + SD v1.4), clean.

    All four are trained on real + BigGAN only; the probes are linear models on frozen
    CLIP ViT-L/14 features, the CNN / ViT fine-tune their last 4 blocks with augmentation.
    """
    rows = []
    for _, r in probe[probe["Experiment"] == "GenImage Authenticity Probe"].iterrows():
        probe_name = {"Logistic Regression": "logistic regression"}.get(r["Model"], r["Model"])
        rows.append({"model": f"CLIP ViT-L/14 + {probe_name} (frozen)",
                     "accuracy": r["Accuracy"], "precision": r["Precision"],
                     "recall": r["Recall"], "F1": r["F1"], "AUROC": r["ROC_AUC"]})
    test = cnn_vit[(cnn_vit["split"] == "test") & (cnn_vit["corruption"] == "clean")]
    for key, name in DETECTORS.items():
        for _, r in test[test["model"] == key].iterrows():
            rows.append({"model": name, "accuracy": r["accuracy"], "precision": r["precision"],
                         "recall": r["recall"], "F1": r["f1"], "AUROC": r["auroc"]})
    return pd.DataFrame(rows)


def robustness_table(cnn_vit: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """F5 detectors under JPEG 50 / 70 and a 20% crop: accuracy, AUROC and recall."""
    rows = []
    for key, name in DETECTORS.items():
        for _, r in cnn_vit[(cnn_vit["model"] == key) & (cnn_vit["split"] == split)].iterrows():
            rows.append({"model": name, "corruption": r["corruption"], "accuracy": r["accuracy"],
                         "AUROC": r["auroc"], "recall": r["recall"]})
    return pd.DataFrame(rows)


def leakage_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """F3 VERITE pair verdicts on originals vs VAE twins: decoupled vs fused, ALR in %."""
    return pd.DataFrame([{"split": r["Split"], "model": r["Model"], "pairs": int(r["Pairs"]),
                          "original acc.": pct(r["Original Accuracy"]),
                          "twin acc.": pct(r["Twin Accuracy"]),
                          "changed": int(r["Changed Verdicts"]), "ALR": pct(r["ALR (%)"] / 100)}
                         for _, r in pairs.iterrows()])


def onnx_table(lat: pd.DataFrame) -> pd.DataFrame:
    """F5 ONNX export: file size, agreement with PyTorch and ms per image (batch 1)."""
    return pd.DataFrame([{"model": DETECTORS.get(r["arch"], r["arch"]),
                          "ONNX MB": f"{r['onnx_mb']:.1f}",
                          "max diff vs PyTorch": f"{r['max_abs_diff']:.1e}",
                          "CPU ms (ONNX Runtime)": f"{r['ort_cpu_ms_mean']:.2f}",
                          "GPU ms (PyTorch)": f"{r['torch_gpu_ms_mean']:.2f}", "GPU": r["gpu"]}
                         for _, r in lat.iterrows()])


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
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, ncol=min(n, 3),
                  loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" keeps a wide legend inside the image
    fig.savefig(path, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure {path.name}")


def reliability_points(df: pd.DataFrame, n_bins: int = 15, min_count: int = 20) -> tuple[list, list]:
    """(mean confidence, accuracy) per bin with at least ``min_count`` original claims.

    Bins with only a few claims are left out of the plot: their accuracy is noise
    (e.g. 0 of 3 correct) and hides the shape of the curve. ECE itself (metrics.py)
    still uses every bin.
    """
    from src.eval.metrics import LABELS, originals

    orig = originals(df)
    probs = orig[[f"p_{lab}" for lab in LABELS]].to_numpy()
    conf = probs.max(axis=1)
    correct = (orig["pred"] == orig["gold"]).to_numpy()
    xs, ys = [], []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if mask.sum() >= min_count:
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
    ax.set_title("Calibration on original test claims\n(seed 42; bins with ≥ 20 claims)",
                 fontsize=9, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure {path.name}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_all(results: Path, tables: Path, figures: Path, gold_paths: list[str]) -> None:
    """Make every table and figure whose source exists (see the module docstring)."""
    def skip(what: str) -> None:
        """Say that an output was skipped because its source is missing."""
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

    nli = [(model, version, ev) for model, version, name in (
               ("mDeBERTa-v3", "v1", "v1_gold_eval.json"),
               ("mDeBERTa-v3", "v2", "gold_eval.json"),
               ("XLM-R-large", "v1", "v1_gold_eval_xlmr.json"))
           if (ev := read_json(results / "R4" / name))]
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
                    "Cited notes: faithfulness and citation accuracy (zero-shot NLI judge).", tables)
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

    summary_path = results / "H5" / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        write_table(main_table(summary), "tab_main",
                    "Verdict head on MultiFC test and its twins (%, mean ± sd, seeds 42/43/44).", tables)
        write_table(main_table(summary, prefix="avt_"), "tab_averitec",
                    "Verdict head on AVeriTeC test (%, mean ± sd, seeds 42/43/44).", tables)
        by = summary.set_index("model")
        metrics = ["macro_f1", "esr", "unrelated_rate", "paraphrase_consistency"]
        series = {name: [by.loc[key, f"{m}_mean"] for m in metrics]
                  for key, name in MAIN_MODELS.items() if key in by.index}
        grouped_bars([MAIN_METRICS[m] for m in metrics], series, "rate (mean of 3 seeds)",
                     "Verdict head on MultiFC test: twin loss and ablations", figures / "fig_main.png")
    else:
        skip("tab_main / tab_averitec / fig_main")

    boot = read_json(results / "H4" / "bootstrap_twin_vs_notwin.json")
    if boot:
        write_table(bootstrap_table(boot), "tab_bootstrap",
                    "Twin loss minus no twin loss, paired bootstrap (1,000 resamples), in points.", tables)
    else:
        skip("tab_bootstrap")

    f1 = {name: ev for name, file in (("rule-based split", "f1_split.json"),
                                      ("Qwen3-8B few-shot", "f1_qwen.json"))
          if (ev := read_json(results / "Y3" / file))}
    if f1:
        write_table(claims_table(f1), "tab_claims",
                    "Claim extraction on 200 held-out messages (match: BGE-M3 cosine >= 0.8).", tables)
    else:
        skip("tab_claims")

    # calibration curves of the H4 seed-42 heads, before and after temperature scaling
    pred_files = {name: results / "H4" / f"preds_{key}_s42.jsonl"
                  for key, name in (("notwin", "no twin loss"), ("notwin_cal", "no twin loss, cal."),
                                    ("twin", "twin loss"), ("twin_cal", "twin loss, cal."))}
    pred_files = {name: path for name, path in pred_files.items() if path.exists()}
    if pred_files and all(Path(p).exists() for p in gold_paths):
        from src.eval.loaders import join_gold, load_jsonl, load_preds
        gold = [r for p in gold_paths for r in load_jsonl(p)]
        curves = {}
        for name, path in pred_files.items():
            df, _ = join_gold(load_preds(path), gold)
            curves[name] = reliability_points(df)
        reliability_figure(curves, figures / "fig_reliability.png")
    else:
        skip("fig_reliability")

    arabic = results / "Y5" / "arabic_E7.csv"
    if arabic.exists():
        write_table(pd.read_csv(arabic), "tab_arabic_E7", "Arabic and mixed-language results (E7).",
                    tables)
    else:
        skip("tab_arabic_E7")

    audit_path = results / "F4" / "audit_E0.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        write_table(audit_table(audit, "multifc"), "tab_audit_E0",
                    "Baselines on MultiFC test and its English twins (E0), in %.", tables)
        write_table(audit_table(audit, "averitec"), "tab_audit_averitec",
                    "Baselines on AVeriTeC test, in %.", tables)
    else:
        skip("tab_audit_E0 / tab_audit_averitec")

    # vision (MAAI901): F3 probe + leakage, F5 detectors, robustness and ONNX
    probe, cnn_vit = results / "F3" / "F3_FINAL_SUMMARY.csv", results / "F5" / "cnn_vit.csv"
    if probe.exists() and cnn_vit.exists():
        cnn = pd.read_csv(cnn_vit)
        write_table(detector_table(pd.read_csv(probe), cnn), "tab_detectors",
                    "AI-image detectors trained on real + BigGAN, tested on real + unseen "
                    "Stable Diffusion v1.4 (clean, seed 42).", tables)
        write_table(robustness_table(cnn), "tab_robustness",
                    "Detectors on the unseen-generator test under JPEG and cropping.", tables)
    else:
        skip("tab_detectors / tab_robustness")
    pairs = results / "F3" / "F3_corrected_pair_ALR_summary.csv"
    if pairs.exists():
        write_table(leakage_table(pd.read_csv(pairs)), "tab_leakage",
                    "VERITE pair verdicts, originals vs VAE twins: accuracy and ALR (%).", tables)
    else:
        skip("tab_leakage")
    onnx = results / "F5" / "onnx_latency.csv"
    if onnx.exists():
        write_table(onnx_table(pd.read_csv(onnx)), "tab_onnx",
                    "ONNX export of the detectors: size, agreement and latency (batch 1, 224x224).",
                    tables)
    else:
        skip("tab_onnx")


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
