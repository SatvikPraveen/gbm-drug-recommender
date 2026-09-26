"""
Figures for the pipeline report. Every function returns the saved path.

Static matplotlib output only; the Streamlit dashboard builds its own
interactive plots from the same result tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from ..config import COLORMAP_SIMILARITY, FIGURE_DPI, FIGURE_FORMAT, FIGURES_DIR, PLOT_STYLE  # noqa: E402

logger = logging.getLogger(__name__)

try:
    plt.style.use(PLOT_STYLE)
except OSError:
    plt.style.use("default")


def _save(fig: plt.Figure, name: str, out_dir: Path = FIGURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)
    return path


def selectivity_volcano(
    summary: pd.DataFrame, name: str = "selectivity_volcano", label_top: int = 12
) -> Path:
    """Mean GBM z-score vs -log10 FDR for every drug; selective drugs highlighted and labelled."""
    df = summary.dropna(subset=["z_mean", "z_q"]).copy()
    df["neglog_q"] = -np.log10(df["z_q"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(8, 6))
    sel = df["gbm_selective"].astype(bool)
    ax.scatter(
        df.loc[~sel, "z_mean"], df.loc[~sel, "neglog_q"], s=14, c="#9aa0a6", alpha=0.7, label="other drugs"
    )
    ax.scatter(
        df.loc[sel, "z_mean"],
        df.loc[sel, "neglog_q"],
        s=28,
        c="#d62728",
        alpha=0.9,
        label="GBM-selective (FDR < 0.05)",
    )
    ax.axhline(-np.log10(0.05), ls="--", lw=1, c="k", alpha=0.5)
    ax.axvline(0, ls=":", lw=1, c="k", alpha=0.5)
    for _, r in df[sel].nsmallest(label_top, "z_mean").iterrows():
        ax.annotate(
            r["drug_name"],
            (r["z_mean"], r["neglog_q"]),
            fontsize=7,
            xytext=(3, 2),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean GDSC z-score across GBM cell lines  (negative = more sensitive than pan-cancer)")
    ax.set_ylabel("-log10 BH-adjusted p  (one-sample t-test vs 0)")
    ax.set_title("GBM-selective sensitivity across screened drugs")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, name)


def benchmark_bars(summary: pd.DataFrame, task: str, metric: str, name: str | None = None) -> Path:
    """Mean ± 95 % CI of one metric for every model under each split strategy."""
    df = summary[(summary["task"] == task) & (summary["metric"] == metric)].copy()
    if df.empty:
        raise ValueError(f"no rows for task={task!r} metric={metric!r}")
    order = df[df["strategy"] == df["strategy"].iloc[0]].sort_values("mean")["model"].tolist()
    strategies = list(dict.fromkeys(df["strategy"]))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.5))
    height = 0.8 / len(strategies)
    for k, strat in enumerate(strategies):
        sub = df[df["strategy"] == strat].set_index("model").reindex(order)
        y = np.arange(len(order)) + (k - (len(strategies) - 1) / 2) * height
        err = np.vstack([sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]]).clip(min=0)
        ax.barh(y, sub["mean"], height=height, xerr=err, label=f"{strat} CV", alpha=0.85, capsize=2)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel(f"{metric} (mean ± 95 % bootstrap CI over folds)")
    ax.set_title(f"{task}: cross-validated {metric}")
    ax.axvline(0, c="k", lw=0.8)
    ax.legend(fontsize=8)
    return _save(fig, name or f"benchmark_{task}_{metric}")


def oof_scatter(oof: pd.DataFrame, task: str, model: str, strategy: str, name: str | None = None) -> Path:
    """Out-of-fold predicted vs observed for one model."""
    df = oof[(oof["task"] == task) & (oof["model"] == model) & (oof["strategy"] == strategy)]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["y_true"], df["y_pred"], s=14, alpha=0.6)
    lo, hi = (
        float(min(df["y_true"].min(), df["y_pred"].min())),
        float(max(df["y_true"].max(), df["y_pred"].max())),
    )
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    rho = df[["y_true", "y_pred"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("observed")
    ax.set_ylabel("out-of-fold prediction")
    ax.set_title(f"{model}\n{task}, {strategy} CV, Spearman ρ = {rho:.2f}", fontsize=10)
    return _save(fig, name or f"oof_{task}_{strategy}")


def y_scramble_hist(scramble: pd.DataFrame, name: str = "y_scrambling") -> Path:
    """Null distribution of the primary metric under label permutation, with the real score marked."""
    tasks = list(dict.fromkeys(scramble["task"]))
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 3.8), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        df = scramble[scramble["task"] == task]
        null = df.loc[df["round"] >= 0, "value"]
        real = df.loc[df["round"] == -1, "value"].iloc[0]
        ax.hist(null, bins=15, color="#9aa0a6", alpha=0.9, label="permuted labels")
        ax.axvline(real, c="#d62728", lw=2, label=f"real = {real:.2f}")
        ax.set_title(f"{task}\n{df['model'].iloc[0]}", fontsize=9)
        ax.set_xlabel(df["metric"].iloc[0])
        ax.legend(fontsize=7)
    return _save(fig, name)


def similarity_heatmap(matrix: pd.DataFrame, title: str, name: str, order: list[str] | None = None) -> Path:
    m = matrix if order is None else matrix.loc[order, order]
    n = len(m)
    fig, ax = plt.subplots(figsize=(max(6, 0.22 * n), max(5, 0.22 * n)))
    sns.heatmap(
        m,
        cmap=COLORMAP_SIMILARITY,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "similarity"},
        ax=ax,
        xticklabels=n <= 80,
        yticklabels=n <= 80,
    )
    ax.tick_params(labelsize=6)
    ax.set_title(title)
    return _save(fig, name)


def similarity_distributions(
    matrices: dict[str, pd.DataFrame], name: str = "similarity_distributions"
) -> Path:
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.2 * len(matrices), 3.5), squeeze=False)
    for ax, (label, m) in zip(axes[0], matrices.items()):
        a = m.to_numpy(dtype=float)
        v = a[np.triu_indices_from(a, k=1)]
        v = v[~np.isnan(v)]
        ax.hist(v, bins=40, color="#4c72b0", alpha=0.9)
        ax.set_title(f"{label} (n pairs = {len(v)})", fontsize=9)
        ax.set_xlabel("similarity")
    return _save(fig, name)


def clustering_scatter(table: pd.DataFrame, summary: pd.DataFrame, name: str = "clustering_umap") -> Path:
    """UMAP embedding coloured by K-means cluster, with GBM-selective drugs outlined."""
    df = table.merge(summary[["drug_name", "gbm_selective", "z_mean"]], on="drug_name", how="left")
    x, y = ("umap_1", "umap_2") if df["umap_1"].notna().any() else ("pca_1", "pca_2")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    sc = axes[0].scatter(df[x], df[y], c=df["kmeans_cluster"], cmap="tab10", s=16, alpha=0.85)
    axes[0].set_title("K-means clusters (descriptor space)")
    axes[0].legend(*sc.legend_elements(), title="cluster", fontsize=7, loc="best")
    sel = df["gbm_selective"].astype(bool)
    axes[1].scatter(
        df.loc[~sel, x],
        df.loc[~sel, y],
        c=df.loc[~sel, "z_mean"],
        cmap="coolwarm",
        s=16,
        alpha=0.7,
        vmin=-1,
        vmax=1,
    )
    axes[1].scatter(
        df.loc[sel, x], df.loc[sel, y], facecolors="none", edgecolors="k", s=60, lw=1.2, label="GBM-selective"
    )
    axes[1].set_title("same embedding coloured by mean GBM z-score")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel(x.upper().replace("_", " "))
        ax.set_ylabel(y.upper().replace("_", " "))
    return _save(fig, name)


def top_drugs_bar(summary: pd.DataFrame, n: int = 25, name: str = "top_gbm_selective_drugs") -> Path:
    df = summary.nsmallest(n, "z_mean").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * n + 1))
    colors = np.where(df["gbm_selective"].astype(bool), "#d62728", "#9aa0a6")
    ax.barh(
        df["drug_name"], -df["z_mean"], xerr=df["z_sd"] / np.sqrt(df["n_cell_lines"]), color=colors, capsize=2
    )
    ax.set_xlabel("-(mean GBM z-score) ± SEM   (higher = more GBM-selective)")
    ax.set_title(f"Top {n} drugs by GBM-selective sensitivity (red = FDR < 0.05)")
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, name)


def pathway_bar(results: pd.DataFrame, n: int = 15, name: str = "pathway_enrichment") -> Path:
    df = results.nsmallest(n, "q").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(df) + 1))
    ax.barh(
        [f"{t[:60]} ({lib.split('_')[0]})" for t, lib in zip(df["term"], df["library"])],
        -np.log10(df["q"].clip(lower=1e-300)),
        color="#4c72b0",
    )
    ax.axvline(-np.log10(0.05), ls="--", c="k", lw=1)
    ax.set_xlabel("-log10 FDR q")
    ax.set_title(
        "Pathways enriched among targets of GBM-selective drugs\n(background: targets of all screened drugs)",
        fontsize=10,
    )
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, name)


def gnn_training_curve(history: dict[str, list[float]], name: str = "gnn_training_history") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (standardised target)")
    ax.set_title("GNN training (final model on all labelled drugs)")
    ax.legend()
    return _save(fig, name)
