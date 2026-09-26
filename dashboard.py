"""
Streamlit dashboard over the pipeline's result tables.

    streamlit run dashboard.py

Every view reads from results/ (run `python main.py` first). The dashboard adds
no analysis of its own, so what it shows is exactly what RESULTS.md reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from gbm_drug import config as cfg

st.set_page_config(page_title="GBM drug response — GDSC", page_icon="🧬", layout="wide")


@st.cache_data
def read_csv(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


@st.cache_data
def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


summary = read_csv(cfg.DRUG_SUMMARY_FILE)
scores = read_csv(cfg.MODEL_RESULTS_DIR / "drug_scores.csv")
bench = read_csv(cfg.BENCHMARK_RESULTS_DIR / "summary.csv")
best = read_csv(cfg.BENCHMARK_RESULTS_DIR / "best_models.csv")
oof = read_csv(cfg.BENCHMARK_RESULTS_DIR / "oof_predictions.csv")
scramble = read_json(cfg.BENCHMARK_RESULTS_DIR / "y_scrambling_summary.json")
pairs = read_csv(cfg.COMBINATION_RESULTS_DIR / "pair_scores.csv")
screen = read_csv(cfg.INTERACTION_RESULTS_DIR / "interaction_screen.csv")
enrich = read_csv(cfg.PATHWAY_RESULTS_DIR / "enrichment_selective_vs_screened.csv")
sim_summary = read_json(cfg.SIMILARITY_RESULTS_DIR / "similarity_summary.json")
meta = read_json(cfg.RESULTS_DIR / "metadata.json")

st.title("GBM drug response from GDSC: structure, selectivity, combinations")
if meta:
    st.caption(
        f"Run {meta.get('timestamp_utc', '')} · commit {str(meta.get('git_commit', ''))[:8]} · GDSC release {cfg.GDSC_RELEASE}"
    )

page = st.sidebar.radio("View", ["Overview", "Drugs", "Benchmark", "Similarity", "Combinations", "Pathways"])

if summary is None:
    st.error("No results found. Run `python main.py` (or `python main.py --quick`) first.")
    st.stop()

if page == "Overview":
    d = meta.get("data", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dose-response curves", d.get("n_curves", len(summary)))
    c2.metric("GBM cell lines", d.get("n_cell_lines", "–"))
    c3.metric("Drugs", d.get("n_drugs", len(summary)))
    c4.metric("GBM-selective drugs (FDR<0.05)", int(summary["gbm_selective"].sum()))
    st.markdown(
        "**Selectivity volcano.** Negative mean z = GBM lines more sensitive than the pan-cancer panel."
    )
    df = summary.dropna(subset=["z_q"]).copy()
    df["-log10 q"] = -df["z_q"].clip(lower=1e-300).apply(lambda v: __import__("math").log10(v))
    fig = px.scatter(
        df,
        x="z_mean",
        y="-log10 q",
        color="gbm_selective",
        hover_name="drug_name",
        hover_data=["putative_target", "pathway_name", "n_cell_lines"],
        color_discrete_map={True: "#d62728", False: "#9aa0a6"},
    )
    fig.add_hline(y=-__import__("math").log10(0.05), line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    if (cfg.RESULTS_DIR / "RESULTS.md").exists():
        with st.expander("Auto-generated results summary (RESULTS.md)"):
            st.markdown((cfg.RESULTS_DIR / "RESULTS.md").read_text())

elif page == "Drugs":
    table = scores if scores is not None else summary
    st.markdown(
        "One row per drug. `oof_pred_*` are out-of-fold predictions from the best model under grouped CV; `novelty_score` is the out-of-fold One-Class SVM score."
    )
    only_sel = st.checkbox("GBM-selective only", value=False)
    q = st.text_input("Filter by drug name / target / pathway")
    view = table[table["gbm_selective"].astype(bool)] if only_sel else table
    if q:
        mask = (
            view[["drug_name", "putative_target", "pathway_name"]]
            .astype(str)
            .apply(lambda s: s.str.contains(q, case=False, na=False))
            .any(axis=1)
        )
        view = view[mask]
    st.dataframe(view, use_container_width=True, height=600)

elif page == "Benchmark":
    if bench is None:
        st.info("Benchmark stage not run.")
    else:
        task = st.selectbox("Task", sorted(bench["task"].unique()))
        metrics = sorted(bench[bench["task"] == task]["metric"].unique())
        metric = st.selectbox(
            "Metric", metrics, index=metrics.index("spearman_rho") if "spearman_rho" in metrics else 0
        )
        df = bench[(bench["task"] == task) & (bench["metric"] == metric)].copy()
        df["err_low"] = df["mean"] - df["ci_low"]
        df["err_high"] = df["ci_high"] - df["mean"]
        fig = px.bar(
            df,
            x="mean",
            y="model",
            color="strategy",
            barmode="group",
            orientation="h",
            error_x="err_high",
            error_x_minus="err_low",
            title=f"{task}: {metric} (mean, 95% bootstrap CI over folds)",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
        if scramble.get(task):
            s = scramble[task]
            st.markdown(
                f"**y-scrambling** ({s['model']}, grouped CV): real {s['metric']} = {s['real']:.3f}, permuted-label mean = {s['null_mean']:.3f}, empirical p = {s['empirical_p']:.3f} ({s['n_rounds']} rounds)."
            )
        if oof is not None and best is not None:
            row = best[(best["task"] == task) & (best["strategy"] == "grouped")]
            if len(row):
                model = row.iloc[0]["model"]
                sub = oof[(oof["task"] == task) & (oof["model"] == model) & (oof["strategy"] == "grouped")]
                st.plotly_chart(
                    px.scatter(
                        sub,
                        x="y_true",
                        y="y_pred",
                        hover_name="drug_name",
                        title=f"Out-of-fold predictions: {model}",
                        trendline=None,
                    ),
                    use_container_width=True,
                )

elif page == "Similarity":
    st.markdown(
        "Candidate drugs = GBM-selective ∪ top-N by selectivity. Three views: Morgan/Tanimoto, MCS-Tanimoto (rdFMCS) and cosine similarity of GNN embeddings trained on GBM selectivity."
    )
    for name in ("tanimoto", "mcs", "gnn"):
        m = read_csv(cfg.SIMILARITY_RESULTS_DIR / f"{name}_candidates.csv", index_col=0)
        if m is None:
            continue
        st.plotly_chart(
            px.imshow(
                m,
                zmin=0,
                zmax=1,
                color_continuous_scale="Viridis",
                title=f"{name} similarity",
                aspect="auto",
                height=650,
            ),
            use_container_width=True,
        )
    if sim_summary.get("mantel"):
        st.markdown("**Mantel tests** (Spearman r, permutation p):")
        st.dataframe(pd.DataFrame([{"pair": k, **v} for k, v in sim_summary["mantel"].items()]))

elif page == "Combinations":
    if pairs is None:
        st.info("Combination stage not run.")
    else:
        st.warning(
            "Hypothesis-generating heuristic built from single-agent GDSC data and annotations. Not a synergy prediction; nothing here is validated against combination screens."
        )
        n = st.slider("Show top N pairs", 10, 100, 25)
        show = pairs[pairs["excluded_reason"].isna()].head(n)
        st.dataframe(
            show[
                [
                    "rank",
                    "drug_a",
                    "drug_b",
                    "total_score",
                    "target_diversity",
                    "pathway_complementarity",
                    "potency",
                    "structural_novelty",
                    "tanimoto",
                    "rationale",
                ]
            ],
            use_container_width=True,
        )
        if screen is not None:
            st.markdown(
                "**Structural screen** of the top pairs (`no_flag` means no rule fired, not that the pair is safe):"
            )
            st.dataframe(screen, use_container_width=True)

elif page == "Pathways":
    if enrich is None or enrich.empty:
        st.info("Pathway stage not run or no gene sets available.")
    else:
        st.markdown(
            "Targets of GBM-selective drugs vs targets of all screened drugs (hypergeometric, BH FDR)."
        )
        sig = enrich[enrich["q"] < cfg.PATHWAY_FDR]
        st.dataframe(sig if len(sig) else enrich.head(30), use_container_width=True)
