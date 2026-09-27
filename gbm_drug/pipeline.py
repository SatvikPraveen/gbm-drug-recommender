"""
Pipeline orchestration: data -> features -> benchmark -> downstream analyses -> figures -> report.

Each stage is a function that reads from and writes to a :class:`Context`. Stages
are idempotent and write their outputs under ``results/``. ``run(options)`` runs the
requested stages in dependency order; ``main.py`` is the CLI.

Stage graph
-----------
data ─ features ─┬─ benchmark ─ final_models ─┬─ similarity ─ combinations
                 ├─ novelty                   │
                 ├─ clustering                └─ figures ─ report
                 └─ pathways
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from . import evaluation as ev
from .combination_therapy import pair_matrix, score_pairs
from .data_processing import load_drug_summary, load_gbm_dose_response
from .drug_interactions import screen_pairs
from .drug_interactions import summarize as summarize_screen
from .feature_extraction import (
    build_feature_table,
    build_fingerprint_matrix,
    descriptor_matrix,
    load_smiles_table,
)
from .models import cluster_drugs, novelty_scores, novelty_table
from .models.zoo import all_models, tabular_models
from .pathway_analysis import drug_gene_map, enrich_libraries, gbm_relevant
from .similarity import gnn_similarity_matrix, mantel_test, mcs_matrix, summarize_matrix, tanimoto_matrix
from .utils import run_metadata, write_json
from .utils import visualization as viz

logger = logging.getLogger(__name__)

TASKS = [
    ev.Task(
        "gbm_selectivity",
        "regression",
        "z_mean",
        "mean GDSC z-score over GBM lines (primary endpoint; lower = more GBM-selective)",
    ),
    ev.Task(
        "gbm_potency", "regression", "ln_ic50_mean", "mean ln IC50 (µM) over GBM lines (lower = more potent)"
    ),
    ev.Task(
        "gbm_selective_class", "classification", "gbm_selective", "GBM-selective at FDR 0.05 and z <= -0.3"
    ),
]
PRIMARY_TASK = TASKS[0]

STAGES = [
    "data",
    "features",
    "benchmark",
    "final_models",
    "novelty",
    "clustering",
    "similarity",
    "combinations",
    "pathways",
    "figures",
    "report",
]


@dataclass
class Options:
    stages: list[str] = field(default_factory=lambda: list(STAGES))
    quick: bool = False
    include_gnn: bool = True
    gnn_repeats: int = 1
    cv_repeats: int = cfg.CV_REPEATS
    scramble_rounds: int = cfg.Y_SCRAMBLE_ROUNDS
    mcs_max_drugs: int = cfg.MCS_MAX_DRUGS
    top_n: int = cfg.TOP_N_DRUGS
    device: str = "auto"
    seed: int = cfg.RANDOM_STATE
    tune: bool = True  # nested hyper-parameter tuning for models that declare a search space

    def __post_init__(self):
        if self.quick:
            self.include_gnn = False
            self.tune = False
            self.cv_repeats = 1
            self.scramble_rounds = 5
            self.mcs_max_drugs = min(self.mcs_max_drugs, 40)


@dataclass
class Context:
    options: Options
    summary: pd.DataFrame | None = None
    dose_response_counts: dict = field(default_factory=dict)
    smiles_table: pd.DataFrame | None = None
    targets: pd.DataFrame | None = None  # one row per drug with structure, merged with summary
    features: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    fold_scores: pd.DataFrame | None = None
    score_summary: pd.DataFrame | None = None
    best: pd.DataFrame | None = None
    oof: pd.DataFrame | None = None
    scramble: pd.DataFrame | None = None
    scramble_p: dict = field(default_factory=dict)
    final_models: dict = field(default_factory=dict)
    gnn: object | None = None
    drug_scores: pd.DataFrame | None = None
    novelty_metrics: dict = field(default_factory=dict)
    cluster_metrics: dict = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)
    similarity: dict = field(default_factory=dict)
    similarity_summary: dict = field(default_factory=dict)
    pairs: pd.DataFrame | None = None
    screen_summary: dict = field(default_factory=dict)
    enrichment: pd.DataFrame | None = None
    pathway_notes: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    @property
    def smiles_list(self) -> list[str]:
        return list(self.targets["smiles_parent"])


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_data(ctx: Context) -> None:
    ctx.summary = load_drug_summary()
    curves = load_gbm_dose_response()
    ctx.dose_response_counts = {
        "n_curves": int(len(curves)),
        "n_cell_lines": int(curves["cell_line"].nunique()),
        "n_drugs": int(curves["drug_name"].nunique()),
        "n_gbm_selective": int(ctx.summary["gbm_selective"].sum()),
        "n_potent": int(ctx.summary["potent"].sum()),
        "datasets": sorted(curves["dataset"].unique().tolist()),
    }
    logger.info("Data: %s", ctx.dose_response_counts)


def stage_features(ctx: Context) -> None:
    table = load_smiles_table()
    ctx.smiles_table = table
    usable = table[table["smiles_parent"].notna() & table["is_small_molecule"].astype(bool)]
    smiles = dict(zip(usable["drug_name"], usable["smiles_parent"]))
    feats = build_feature_table(smiles)
    feats = feats.merge(
        usable[["drug_name", "inchikey_parent", "pubchem_cid", "has_metal"]], on="drug_name", how="left"
    )
    targets = ctx.summary.merge(feats, on="drug_name", how="inner").reset_index(drop=True)
    targets = targets.rename(columns={"smiles": "smiles_parent"})
    ctx.targets = targets

    X_desc = descriptor_matrix(targets)
    X_morgan, _ = build_fingerprint_matrix(dict(zip(targets["drug_name"], targets["smiles_parent"])))
    X_pathway = pathway_onehot(targets["pathway_name"])
    ctx.features = {
        "descriptors": X_desc,
        "morgan": X_morgan,
        "smiles": list(targets["smiles_parent"]),
        "pathway": X_pathway,
        "descriptors_pathway": np.hstack([X_desc, X_pathway]),
    }
    ctx.groups = {
        "grouped": ev.connectivity_groups(targets["inchikey_parent"].tolist()),
        "scaffold": ev.scaffold_groups(targets["scaffold"].tolist()),
    }
    cfg.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    targets.to_csv(cfg.FEATURES_FILE, index=False)
    logger.info(
        "Features: %d drugs with structures (%d unique molecules, %d scaffolds); %d GBM-selective among them",
        len(targets),
        len(set(ctx.groups["grouped"])),
        len(set(ctx.groups["scaffold"])),
        int(targets["gbm_selective"].sum()),
    )


def pathway_onehot(pathways: pd.Series) -> np.ndarray:
    """One-hot encoding of GDSC PATHWAY_NAME (unknown/'Other' become their own column)."""
    labels = pathways.fillna("Unclassified").astype(str).str.strip()
    return pd.get_dummies(labels, dtype=np.uint8).to_numpy()


def _models(ctx: Context) -> list[ev.ModelSpec]:
    return all_models(
        include_gnn=ctx.options.include_gnn, gnn_repeats=ctx.options.gnn_repeats, device=ctx.options.device
    )


def stage_benchmark(ctx: Context) -> None:
    out = cfg.BENCHMARK_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    specs = _models(ctx)
    fold_scores, oof = ev.run_benchmark(
        TASKS,
        specs,
        ctx.features,
        ctx.targets,
        ctx.groups,
        n_splits=cfg.CV_FOLDS,
        n_repeats=ctx.options.cv_repeats,
        random_state=ctx.options.seed,
    )
    summary = ev.summarize_scores(fold_scores)
    best = ev.best_models(summary, TASKS)
    fold_scores.to_csv(out / "fold_scores.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    best.to_csv(out / "best_models.csv", index=False)
    oof.to_csv(out / "oof_predictions.csv", index=False)
    ctx.fold_scores, ctx.score_summary, ctx.best, ctx.oof = fold_scores, summary, best, oof

    # y-scrambling for the best *tabular* model per task under grouped CV (GNNs are too slow to permute).
    tab = {s.name: s for s in tabular_models()}
    frames = []
    for task in TASKS:
        cand = summary[
            (summary["task"] == task.name)
            & (summary["strategy"] == "grouped")
            & (summary["metric"] == ev.PRIMARY_METRIC[task.kind])
            & summary["model"].isin(tab)
        ]
        if cand.empty:
            continue
        spec = tab[cand.sort_values("mean").iloc[-1]["model"]]
        df = ev.y_scramble(
            task,
            spec,
            ctx.features,
            ctx.targets,
            ctx.groups["grouped"],
            "grouped",
            n_rounds=ctx.options.scramble_rounds,
            random_state=ctx.options.seed,
        )
        ctx.scramble_p[task.name] = {
            "model": spec.name,
            "metric": ev.PRIMARY_METRIC[task.kind],
            "empirical_p": df.attrs["empirical_p"],
            "real": float(df.loc[df["round"] == -1, "value"].iloc[0]),
            "null_mean": float(df.loc[df["round"] >= 0, "value"].mean()),
            "n_rounds": ctx.options.scramble_rounds,
        }
        frames.append(df)
    if frames:
        ctx.scramble = pd.concat(frames, ignore_index=True)
        ctx.scramble.to_csv(out / "y_scrambling.csv", index=False)
        write_json(ctx.scramble_p, out / "y_scrambling_summary.json")


def _load_benchmark_artifacts(ctx: Context) -> None:
    """Populate ctx from results/benchmark if the benchmark stage did not run in this process."""
    if ctx.best is not None:
        return
    out = cfg.BENCHMARK_RESULTS_DIR
    needed = [out / "summary.csv", out / "best_models.csv", out / "oof_predictions.csv"]
    if not all(p.exists() for p in needed):
        raise FileNotFoundError("benchmark artifacts missing; run `python main.py --stages benchmark` first")
    ctx.score_summary = pd.read_csv(needed[0])
    ctx.best = pd.read_csv(needed[1])
    ctx.oof = pd.read_csv(needed[2])
    if (out / "fold_scores.csv").exists():
        ctx.fold_scores = pd.read_csv(out / "fold_scores.csv")
    if (out / "y_scrambling.csv").exists():
        ctx.scramble = pd.read_csv(out / "y_scrambling.csv")
    ctx.scramble_p = load_json(out / "y_scrambling_summary.json")
    logger.info("Loaded benchmark artifacts from %s", out)


def _load_saved_gnn(ctx: Context) -> None:
    """Reuse the final GNN saved by an earlier final_models stage, if any."""
    if ctx.gnn is not None or not ctx.options.include_gnn:
        return
    path = cfg.MODEL_RESULTS_DIR / "gnn_gcn_gbm_selectivity.pt"
    if path.exists():
        from .models.gnn_model import GNNDrugPredictor

        ctx.gnn = GNNDrugPredictor.load(path, device=ctx.options.device)
        logger.info("Loaded saved GNN from %s", path.name)


def stage_final_models(ctx: Context) -> None:
    out = cfg.MODEL_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    _load_benchmark_artifacts(ctx)
    specs = {s.name: s for s in _models(ctx)}
    for task in TASKS:
        row = ctx.best[(ctx.best["task"] == task.name) & (ctx.best["strategy"] == "grouped")]
        if row.empty:
            continue
        name = row.iloc[0]["model"]
        spec = specs.get(name)
        if spec is None or spec.features == "smiles":
            # best model is a GNN; keep a tabular fallback for downstream scoring
            tab = ctx.score_summary[
                (ctx.score_summary["task"] == task.name)
                & (ctx.score_summary["strategy"] == "grouped")
                & (ctx.score_summary["metric"] == ev.PRIMARY_METRIC[task.kind])
                & (ctx.score_summary["features"] != "smiles")
                & (ctx.score_summary["model"] != "Baseline")
            ]
            spec = specs[tab.sort_values("mean").iloc[-1]["model"]]
        model = ev.fit_final_model(spec, task, ctx.features, ctx.targets)
        ctx.final_models[task.name] = (spec, model)
        joblib.dump(
            {"spec_name": spec.name, "features": spec.features, "task": task.name, "model": model},
            out / f"final_{task.name}.joblib",
        )
        logger.info("Final model for %s: %s", task.name, spec.name)

    if ctx.options.include_gnn:
        from .models.gnn_model import GNNDrugPredictor

        gnn = GNNDrugPredictor(task="regression", device=ctx.options.device, random_state=ctx.options.seed)
        gnn.fit(ctx.smiles_list, ctx.targets[PRIMARY_TASK.target].to_numpy(dtype=float))
        gnn.save(out / "gnn_gcn_gbm_selectivity.pt")
        ctx.gnn = gnn
        write_json(
            {"n_epochs": gnn.n_epochs_, "best_val_loss": gnn.best_val_loss_, "history": gnn.history_},
            out / "gnn_training_history.json",
        )

    # Drug score table: summary + OOF predictions (grouped CV, repeat 0) for each task's best model.
    scores = ctx.targets[
        [
            "drug_name",
            "putative_target",
            "pathway_name",
            "n_cell_lines",
            "z_mean",
            "z_q",
            "ln_ic50_mean",
            "ic50_um_geomean",
            "gbm_selective",
            "potent",
            "scaffold",
            "inchikey_parent",
        ]
    ].copy()
    for task in TASKS:
        if task.name not in ctx.final_models:
            continue
        spec, _ = ctx.final_models[task.name]
        oof = ctx.oof[
            (ctx.oof["task"] == task.name)
            & (ctx.oof["strategy"] == "grouped")
            & (ctx.oof["model"] == spec.name)
        ][["drug_name", "y_pred"]]
        scores = scores.merge(
            oof.rename(columns={"y_pred": f"oof_pred_{task.name}"}), on="drug_name", how="left"
        )
    scores = scores.sort_values("z_mean").reset_index(drop=True)
    scores.to_csv(out / "drug_scores.csv", index=False)
    ctx.drug_scores = scores


def stage_novelty(ctx: Context) -> None:
    out = cfg.MODEL_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    positive = ctx.targets["gbm_selective"].astype(bool).to_numpy()
    if positive.sum() < cfg.CV_FOLDS:
        logger.warning(
            "Too few GBM-selective drugs (%d) for one-class novelty scoring; skipping", positive.sum()
        )
        return
    scores, metrics = novelty_scores(
        ctx.features["descriptors"], positive, ctx.groups["grouped"], random_state=ctx.options.seed
    )
    table = novelty_table(ctx.targets["drug_name"], scores, positive)
    table.to_csv(out / "novelty_scores.csv", index=False)
    write_json(metrics, out / "novelty_metrics.json")
    ctx.novelty_metrics = metrics
    if ctx.drug_scores is not None:
        ctx.drug_scores = ctx.drug_scores.merge(
            table[["drug_name", "novelty_score"]], on="drug_name", how="left"
        )
        ctx.drug_scores.to_csv(out / "drug_scores.csv", index=False)


def stage_clustering(ctx: Context) -> None:
    out = cfg.CLUSTERING_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    table, metrics, k_table = cluster_drugs(
        ctx.features["descriptors"], ctx.targets["drug_name"], random_state=ctx.options.seed
    )
    table.to_csv(out / "clusters.csv", index=False)
    k_table.to_csv(out / "silhouette_by_k.csv", index=False)
    write_json(metrics, out / "metrics.json")
    ctx.cluster_metrics = metrics


def _candidate_set(ctx: Context) -> list[str]:
    t = ctx.targets
    top = t.nsmallest(ctx.options.top_n, "z_mean")["drug_name"].tolist()
    sel = t[t["gbm_selective"].astype(bool)]["drug_name"].tolist()
    cand = list(dict.fromkeys(sel + top))
    return cand[: ctx.options.mcs_max_drugs]


def stage_similarity(ctx: Context) -> None:
    out = cfg.SIMILARITY_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    _load_saved_gnn(ctx)
    smiles_all = dict(zip(ctx.targets["drug_name"], ctx.targets["smiles_parent"]))
    ctx.candidates = _candidate_set(ctx)
    (out / "candidates.txt").write_text("\n".join(ctx.candidates) + "\n")
    smiles_cand = {d: smiles_all[d] for d in ctx.candidates}

    tani_all = tanimoto_matrix(smiles_all)
    tani_all.to_csv(out / "tanimoto_all.csv")
    sims = {"tanimoto": tani_all.loc[ctx.candidates, ctx.candidates]}
    sims["tanimoto"].to_csv(out / "tanimoto_candidates.csv")

    sims["mcs"] = mcs_matrix(smiles_cand, timeout=cfg.MCS_TIMEOUT_SECONDS)
    sims["mcs"].to_csv(out / "mcs_candidates.csv")

    if ctx.gnn is not None:
        sims["gnn"] = gnn_similarity_matrix(ctx.gnn, smiles_cand)
        sims["gnn"].to_csv(out / "gnn_candidates.csv")

    summary = {name: summarize_matrix(m, cfg.TANIMOTO_THRESHOLD) for name, m in sims.items()}
    summary["all_drugs_tanimoto"] = summarize_matrix(tani_all, cfg.TANIMOTO_THRESHOLD)
    names = list(sims)
    summary["mantel"] = {
        f"{a}_vs_{b}": mantel_test(sims[a], sims[b], seed=ctx.options.seed)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    write_json(summary, out / "similarity_summary.json")
    ctx.similarity = {"all_tanimoto": tani_all, **sims}
    ctx.similarity_summary = summary


def stage_combinations(ctx: Context) -> None:
    out = cfg.COMBINATION_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    if not ctx.candidates:
        ctx.candidates = _candidate_set(ctx)
    cand = ctx.targets[ctx.targets["drug_name"].isin(ctx.candidates)]
    tani = ctx.similarity.get("all_tanimoto")
    if tani is None:
        tani = tanimoto_matrix(dict(zip(cand["drug_name"], cand["smiles_parent"])))
    pairs = score_pairs(cand, tani)
    pairs.to_csv(out / "pair_scores.csv", index=False)
    pair_matrix(pairs).to_csv(out / "pair_matrix.csv")
    ctx.pairs = pairs

    smiles = dict(zip(ctx.targets["drug_name"], ctx.targets["smiles_parent"]))
    top_pairs = pairs[pairs["excluded_reason"].isna()].head(100)
    screened = screen_pairs(list(zip(top_pairs["drug_a"], top_pairs["drug_b"])), smiles)
    screened.to_csv(
        cfg.INTERACTION_RESULTS_DIR / "interaction_screen.csv", index=False
    ) if cfg.INTERACTION_RESULTS_DIR.mkdir(parents=True, exist_ok=True) is None else None
    ctx.screen_summary = summarize_screen(screened)
    write_json(ctx.screen_summary, cfg.INTERACTION_RESULTS_DIR / "interaction_summary.json")


def stage_pathways(ctx: Context) -> None:
    out = cfg.PATHWAY_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    genes, unmapped = drug_gene_map(ctx.summary)  # all 542 drugs: targets do not need structures
    pd.DataFrame([{"drug_name": d, "genes": ";".join(sorted(g))} for d, g in genes.items()]).to_csv(
        out / "drug_gene_map.csv", index=False
    )
    pd.DataFrame([{"drug_name": d, "unmapped": ";".join(sorted(u))} for d, u in unmapped.items()]).to_csv(
        out / "unmapped_targets.csv", index=False
    )

    selective = ctx.summary[ctx.summary["gbm_selective"].astype(bool)]["drug_name"]
    query = set().union(*(genes[d] for d in selective))
    background = set().union(*genes.values())
    ctx.pathway_notes = {
        "n_query_genes": len(query),
        "n_background_genes": len(background),
        "n_selective_drugs": int(len(selective)),
        "n_drugs_with_mapped_targets": sum(1 for g in genes.values() if g),
        "n_drugs_with_unmapped_tokens": len(unmapped),
    }
    try:
        res = enrich_libraries(query, background, cfg.GENESET_LIBRARIES, cfg.GENESET_DIR)
    except Exception as exc:  # network unavailable or library changed
        logger.warning("Pathway enrichment skipped: %s", exc)
        ctx.pathway_notes["error"] = str(exc)
        write_json(ctx.pathway_notes, out / "enrichment_notes.json")
        return
    res.to_csv(out / "enrichment_selective_vs_screened.csv", index=False)
    gbm_relevant(res).to_csv(out / "enrichment_gbm_relevant.csv", index=False)
    ctx.enrichment = res
    ctx.pathway_notes["n_significant"] = int((res["q"] < cfg.PATHWAY_FDR).sum()) if not res.empty else 0
    write_json(ctx.pathway_notes, out / "enrichment_notes.json")


def stage_figures(ctx: Context) -> None:
    viz.selectivity_volcano(ctx.summary)
    viz.top_drugs_bar(ctx.summary)
    if ctx.score_summary is not None:
        for task in TASKS:
            viz.benchmark_bars(ctx.score_summary, task.name, ev.PRIMARY_METRIC[task.kind])
        if ctx.best is not None and ctx.oof is not None:
            row = ctx.best[(ctx.best["task"] == PRIMARY_TASK.name) & (ctx.best["strategy"] == "grouped")]
            if not row.empty:
                viz.oof_scatter(ctx.oof, PRIMARY_TASK.name, row.iloc[0]["model"], "grouped")
    if ctx.scramble is not None:
        viz.y_scramble_hist(ctx.scramble)
    if ctx.gnn is not None:
        viz.gnn_training_curve(ctx.gnn.history_)
    if ctx.similarity:
        order = ctx.candidates
        for name in ("tanimoto", "mcs", "gnn"):
            if name in ctx.similarity:
                viz.similarity_heatmap(
                    ctx.similarity[name],
                    f"{name.upper()} similarity, candidate drugs",
                    f"{name}_similarity_candidates",
                    order,
                )
        viz.similarity_distributions({k: v for k, v in ctx.similarity.items() if k != "all_tanimoto"})
    clusters = cfg.CLUSTERING_RESULTS_DIR / "clusters.csv"
    if clusters.exists():
        viz.clustering_scatter(pd.read_csv(clusters), ctx.summary)
    if ctx.enrichment is not None and not ctx.enrichment.empty:
        viz.pathway_bar(ctx.enrichment)


def stage_report(ctx: Context) -> None:
    from .reporting import write_results_markdown

    # Keep timings of stages that ran in an earlier process (e.g. `--stages report` after a full run).
    previous = load_json(cfg.RESULTS_DIR / "metadata.json").get("timings_seconds", {})
    ctx.timings = {**{k: v for k, v in previous.items() if k not in ctx.timings}, **ctx.timings}
    meta = run_metadata(
        {
            "options": {
                k: (list(v) if isinstance(v, (set, tuple)) else v) for k, v in vars(ctx.options).items()
            },
            "timings_seconds": ctx.timings,
            "data": ctx.dose_response_counts,
        }
    )
    write_json(meta, cfg.RESULTS_DIR / "metadata.json")
    write_results_markdown(ctx, cfg.RESULTS_DIR / "RESULTS.md")


STAGE_FUNCS = {
    "data": stage_data,
    "features": stage_features,
    "benchmark": stage_benchmark,
    "final_models": stage_final_models,
    "novelty": stage_novelty,
    "clustering": stage_clustering,
    "similarity": stage_similarity,
    "combinations": stage_combinations,
    "pathways": stage_pathways,
    "figures": stage_figures,
    "report": stage_report,
}

# Stages that must run (in this process) before a given stage can run.
REQUIRES = {
    "features": ["data"],
    "benchmark": ["features"],
    "final_models": ["features"],  # loads benchmark artifacts from disk if the stage did not run
    "novelty": ["features"],
    "clustering": ["features"],
    "similarity": ["features"],
    "combinations": ["features"],
    "pathways": ["data"],
    "figures": ["data"],
    "report": ["data"],
}


def resolve_stages(requested: list[str]) -> list[str]:
    wanted: set[str] = set()

    def add(s: str) -> None:
        if s in wanted:
            return
        for dep in REQUIRES.get(s, []):
            add(dep)
        wanted.add(s)

    for s in requested:
        if s not in STAGE_FUNCS:
            raise ValueError(f"unknown stage {s!r}; choose from {STAGES}")
        add(s)
    return [s for s in STAGES if s in wanted]


def run(options: Options) -> Context:
    cfg.ensure_directories()
    ctx = Context(options=options)
    for stage in resolve_stages(options.stages):
        logger.info("=" * 70)
        logger.info("STAGE %s", stage.upper())
        logger.info("=" * 70)
        t0 = time.perf_counter()
        STAGE_FUNCS[stage](ctx)
        ctx.timings[stage] = round(time.perf_counter() - t0, 1)
        logger.info("stage %s finished in %.1fs", stage, ctx.timings[stage])
    return ctx


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}
