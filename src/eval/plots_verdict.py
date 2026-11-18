"""Report figures for the verdict-head experiments (H4/H5), PNG at 300 dpi.

What it does
    1. fig_lambda_grid.png: the H4 λ grid as two heatmaps (validation twin score and
       validation macro-F1 for every λ_inv × λ_sens), the chosen cell outlined.
    2. fig_ablations.png: H5 ablations on MultiFC test, one small panel per metric
       (macro-F1, ESR, unrelated rate, paraphrase consistency), mean ± sd over seeds.
    3. fig_entail_shortcut.png: how well zero-shot NLI separates true from false
       claims: share of MultiFC val claims whose best snippet entail score is at least
       x, for supported vs contradicted claims (the similarity-over-factuality finding).
    Rathi's ``src/eval/figures.py`` (R8) makes the shared data figures; this file
    only covers Himasri's experiments. Every number comes from a file in results/
    or features/.

Inputs
    results/H4/grid_summary.csv, results/H4/selected_lambda.json,
    results/H5/summary.csv, features/nli_zeroshot/multifc_val.parquet,
    data/processed/en/multifc_val.jsonl

Outputs
    reports/figures/fig_lambda_grid.png, fig_ablations.png, fig_entail_shortcut.png

How to run
    python -m src.eval.plots_verdict
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("reports/figures")
# Validated categorical palette (dataviz reference instance, light mode), fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#1f2328", "#5b6475", "#e3e1da", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax) -> None:
    """Recessive axes: no top/right spines, light grid, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(INK)


def lambda_grid(grid_csv: str, selected_json: str, out: Path) -> Path:
    """Two heatmaps of the λ grid with the chosen setting outlined."""
    from matplotlib.colors import LinearSegmentedColormap

    df = pd.read_csv(grid_csv)
    chosen = json.loads(Path(selected_json).read_text())["chosen"]
    inv = sorted(df["lambda_inv"].unique())
    sens = sorted(df["lambda_sens"].unique())
    cmap = LinearSegmentedColormap.from_list("blue", SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    for ax, col, title in ((axes[0], "val_twin_score", "Validation twin score"),
                           (axes[1], "val_macro_f1", "Validation macro-F1")):
        mat = df.pivot(index="lambda_inv", columns="lambda_sens", values=col).loc[inv, sens].to_numpy()
        ax.imshow(mat, cmap=cmap, aspect="auto")
        mid = (np.nanmin(mat) + np.nanmax(mat)) / 2
        for i in range(len(inv)):
            for j in range(len(sens)):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if mat[i, j] > mid else INK)
        ci, cj = inv.index(chosen["lambda_inv"]), sens.index(chosen["lambda_sens"])
        ax.add_patch(plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2.5))
        ax.set_xticks(range(len(sens)), [f"{s:g}" for s in sens])
        ax.set_yticks(range(len(inv)), [f"{v:g}" for v in inv])
        ax.set_xlabel("λ_sens (push _B → contradicted, _C → unverified)", fontsize=9, color=MUTED)
        ax.set_ylabel("λ_inv (keep the answer on _P/_A/_M)", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=11, loc="left")
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("λ grid, seed 42. Outlined: chosen setting (best twin score within 1 F1 point of λ = 0)",
                 fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    path = out / "fig_lambda_grid.png"
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    return path


def ablations(summary_csv: str, out: Path) -> Path:
    """One horizontal-bar panel per metric, models on the y-axis, values labelled."""
    s = pd.read_csv(summary_csv, index_col=0)
    models = [("twin", "Full model", BLUE), ("notwin", "− twin loss", ORANGE),
              ("twin_norel", "− reranker", AQUA), ("twin_noent", "− entity match", YELLOW)]
    metrics = [("macro_f1", "Macro-F1"), ("esr", "Evidence sensitivity (ESR)"),
               ("unrelated_rate", "Unrelated rate"), ("paraphrase_consistency", "Paraphrase consistency")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), facecolor=SURFACE, sharey=True)
    for ax, (m, title) in zip(axes, metrics):
        _style(ax)
        y = np.arange(len(models))[::-1]
        means = [s.loc[k, f"{m}_mean"] * 100 for k, _, _ in models]
        sds = [s.loc[k, f"{m}_std"] * 100 for k, _, _ in models]
        ax.barh(y, means, height=0.62, color=[c for _, _, c in models],
                xerr=sds, error_kw={"ecolor": MUTED, "lw": 1, "capsize": 2})
        for yi, v, sd in zip(y, means, sds):
            ax.text(v + sd + 3, yi, f"{v:.1f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, [lab for _, lab, _ in models])
        ax.set_xlim(0, 118 if m != "macro_f1" else 50)
        ax.set_title(title, fontsize=10, loc="left")
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Ablations, MultiFC test + twins (%, mean ± sd over seeds 42/43/44, zero-shot NLI)",
                 fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    path = out / "fig_ablations.png"
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    return path


def entail_shortcut(nli_parquet: str, records_jsonl: str, out: Path) -> Path:
    """Survival curves of each claim's best snippet entail score, supported vs contradicted."""
    nli = pd.read_parquet(nli_parquet)
    labels = {}
    with open(records_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            labels[rec["id"]] = rec["label"]
    best = nli.groupby("id")["p_entail"].max()
    xs = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    _style(ax)
    for lab, color, name in (("supported", BLUE, "True claims (supported)"),
                             ("contradicted", ORANGE, "False claims (contradicted)")):
        vals = best[best.index.map(labels) == lab].to_numpy()
        share = [(vals >= x).mean() * 100 for x in xs]
        at = 0.9
        v = (vals >= at).mean() * 100
        ax.plot(xs, share, color=color, lw=2, label=f"{name}: {v:.0f}% at ≥ 0.9")
        ax.plot([at], [v], "o", color=color, ms=7, mec=SURFACE, mew=2)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_xlabel("Best snippet's zero-shot entail score (threshold)", fontsize=9, color=MUTED)
    ax.set_ylabel("% of claims at or above threshold", fontsize=9, color=MUTED)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Zero-shot NLI barely separates true from false claims (MultiFC val)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    path = out / "fig_entail_shortcut.png"
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    """Write all three figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (lambda_grid("results/H4/grid_summary.csv", "results/H4/selected_lambda.json", OUT),
              ablations("results/H5/summary.csv", OUT),
              entail_shortcut("features/nli_zeroshot/multifc_val.parquet",
                              "data/processed/en/multifc_val.jsonl", OUT)):
        print("wrote", p)


if __name__ == "__main__":
    main()
