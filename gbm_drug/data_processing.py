"""
GDSC data processing: from the raw release files to a per-drug GBM response table.

Pipeline
--------
1. ``load_gdsc_raw``       read the GDSC1/GDSC2 fitted dose-response workbooks.
2. ``build_gbm_subset``    keep curves from cell lines GDSC labels ``TCGA_DESC == "GBM"``,
                           standardise column names, derive IC50 in µM and an
                           in-range flag. One row per (dataset, drug_id, cell line).
3. ``summarize_drugs``     collapse to one row per drug: potency (mean ln IC50 over
                           GBM lines), GBM selectivity (mean GDSC z-score, one-sample
                           t-test vs 0, BH-FDR), targets and pathway from GDSC's own
                           annotation, and the derived labels used downstream.

Why z-scores
------------
GDSC's ``Z_SCORE`` standardises each drug's ln IC50 across every screened cell line.
A negative mean over GBM lines therefore means GBM is *more sensitive than the
pan-cancer average* to that drug, which is the question a GBM-specific
recommender should ask. Raw IC50 thresholds (e.g. "< 10 µM") conflate general
cytotoxicity with GBM-specific activity and depend on the tested concentration
range, so they are kept only as a secondary label.

Pseudo-replication
------------------
Many drugs were screened in both GDSC1 and GDSC2 (and some twice within one
release). Curves are averaged within (drug, cell line) *before* the t-test so
each GBM cell line contributes one observation per drug.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import (
    DRUG_SUMMARY_FILE,
    GBM_DOSE_RESPONSE_FILE,
    GBM_TCGA_LABEL,
    GDSC1_FILE,
    GDSC2_FILE,
    POTENCY_LN_IC50_THRESHOLD,
    SELECTIVITY_FDR,
    SELECTIVITY_MIN_CELL_LINES,
    SELECTIVITY_Z_EFFECT,
)

logger = logging.getLogger(__name__)

# GDSC column -> snake_case name used throughout the package.
COLUMN_MAP = {
    "DATASET": "dataset",
    "COSMIC_ID": "cosmic_id",
    "CELL_LINE_NAME": "cell_line",
    "SANGER_MODEL_ID": "sanger_model_id",
    "TCGA_DESC": "tcga_desc",
    "DRUG_ID": "drug_id",
    "DRUG_NAME": "drug_name",
    "PUTATIVE_TARGET": "putative_target",
    "PATHWAY_NAME": "pathway_name",
    "MIN_CONC": "min_conc_um",
    "MAX_CONC": "max_conc_um",
    "LN_IC50": "ln_ic50",
    "AUC": "auc",
    "RMSE": "rmse",
    "Z_SCORE": "z_score",
}

REQUIRED_RAW_COLUMNS = (
    "DATASET",
    "CELL_LINE_NAME",
    "TCGA_DESC",
    "DRUG_ID",
    "DRUG_NAME",
    "LN_IC50",
    "AUC",
    "Z_SCORE",
)

SUBSET_COLUMNS = [
    "dataset",
    "drug_id",
    "drug_name",
    "putative_target",
    "pathway_name",
    "cosmic_id",
    "sanger_model_id",
    "cell_line",
    "tcga_desc",
    "min_conc_um",
    "max_conc_um",
    "ln_ic50",
    "ic50_um",
    "ic50_within_range",
    "auc",
    "rmse",
    "z_score",
]


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------


def load_gdsc_raw(paths: Iterable[Path] = (GDSC1_FILE, GDSC2_FILE)) -> pd.DataFrame:
    """Read and concatenate GDSC fitted dose-response workbooks (xlsx or csv)."""
    frames = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python scripts/download_gdsc.py` first.")
        logger.info("Reading %s", path.name)
        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{path.name} is missing expected GDSC columns: {missing}")
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d curves from %d file(s)", len(raw), len(frames))
    return raw


# ---------------------------------------------------------------------------
# GBM subset
# ---------------------------------------------------------------------------


def _normalise_drug_name(name: str) -> str:
    return " ".join(str(name).split())


def build_gbm_subset(raw: pd.DataFrame, tcga_label: str = GBM_TCGA_LABEL) -> pd.DataFrame:
    """
    Filter raw GDSC curves to one tumour type and standardise columns.

    Returns one row per (dataset, drug_id, cell_line) with IC50 in µM and a flag
    for whether the fitted IC50 lies within the tested concentration range
    (IC50s above ``MAX_CONC`` are extrapolations of the curve fit).
    """
    subset = raw.loc[raw["TCGA_DESC"] == tcga_label].copy()
    if subset.empty:
        raise ValueError(f"No curves with TCGA_DESC == {tcga_label!r}; check the input files.")

    subset = subset.rename(columns=COLUMN_MAP)
    subset["drug_name"] = subset["drug_name"].map(_normalise_drug_name)
    subset["ln_ic50"] = subset["ln_ic50"].astype(float)
    subset["ic50_um"] = np.exp(subset["ln_ic50"])
    if "max_conc_um" in subset.columns:
        subset["ic50_within_range"] = subset["ln_ic50"] <= np.log(subset["max_conc_um"].astype(float))
    else:
        subset["max_conc_um"] = np.nan
        subset["min_conc_um"] = np.nan
        subset["ic50_within_range"] = pd.NA

    for col in SUBSET_COLUMNS:
        if col not in subset.columns:
            subset[col] = pd.NA

    subset = (
        subset[SUBSET_COLUMNS]
        .sort_values(["drug_name", "dataset", "drug_id", "cell_line"])
        .reset_index(drop=True)
    )
    logger.info(
        "%s subset: %d curves, %d cell lines, %d drugs",
        tcga_label,
        len(subset),
        subset["cell_line"].nunique(),
        subset["drug_name"].nunique(),
    )
    return subset


# ---------------------------------------------------------------------------
# Per-drug summary
# ---------------------------------------------------------------------------


def _mode_or_first(values: pd.Series) -> str | None:
    values = values.dropna().astype(str)
    if values.empty:
        return None
    return values.mode().iloc[0]


def _join_unique(values: pd.Series) -> str:
    return ";".join(sorted({str(v) for v in values.dropna()}))


def summarize_drugs(
    subset: pd.DataFrame,
    fdr: float = SELECTIVITY_FDR,
    min_cell_lines: int = SELECTIVITY_MIN_CELL_LINES,
    z_effect: float = SELECTIVITY_Z_EFFECT,
    potency_threshold: float = POTENCY_LN_IC50_THRESHOLD,
) -> pd.DataFrame:
    """
    Collapse GBM curves to one row per drug with potency and selectivity statistics.

    Columns
    -------
    n_curves, n_cell_lines, datasets, drug_ids
    ln_ic50_mean / _median / _sd   potency across GBM lines (ln µM; lower = more potent)
    ic50_um_geomean                exp(ln_ic50_mean)
    auc_mean                       mean area under the dose-response curve (lower = more sensitive)
    frac_ic50_within_range         share of curves whose IC50 lies inside the tested range
    z_mean, z_sd                   GDSC z-score statistics (negative = GBM more sensitive than average)
    z_t, z_p, z_q                  one-sample t-test of z-scores vs 0 and BH-adjusted p
    gbm_selective                  z_q < fdr and z_mean <= z_effect and n_cell_lines >= min_cell_lines
    potent                         ln_ic50_mean < potency_threshold
    putative_target, pathway_name  GDSC's annotation (mode across curves)
    """
    # Average duplicate screens of the same drug on the same cell line first so
    # the t-test sees each GBM cell line once per drug.
    per_line = (
        subset.groupby(["drug_name", "cell_line"], as_index=False)
        .agg(ln_ic50=("ln_ic50", "mean"), auc=("auc", "mean"), z_score=("z_score", "mean"))
        .dropna(subset=["ln_ic50"])
    )

    def _drug_stats(group: pd.DataFrame) -> pd.Series:
        z = group["z_score"].dropna().to_numpy()
        n = len(z)
        if n >= 2 and np.std(z, ddof=1) > 0:
            t_stat, p_val = stats.ttest_1samp(z, popmean=0.0)
        else:
            t_stat, p_val = np.nan, np.nan
        return pd.Series(
            {
                "n_cell_lines": int(group["cell_line"].nunique()),
                "ln_ic50_mean": group["ln_ic50"].mean(),
                "ln_ic50_median": group["ln_ic50"].median(),
                "ln_ic50_sd": group["ln_ic50"].std(ddof=1),
                "auc_mean": group["auc"].mean(),
                "z_mean": z.mean() if n else np.nan,
                "z_sd": np.std(z, ddof=1) if n >= 2 else np.nan,
                "z_t": t_stat,
                "z_p": p_val,
            }
        )

    summary = per_line.groupby("drug_name").apply(_drug_stats, include_groups=False).reset_index()

    meta = (
        subset.groupby("drug_name")
        .agg(
            n_curves=("ln_ic50", "size"),
            datasets=("dataset", _join_unique),
            drug_ids=("drug_id", _join_unique),
            putative_target=("putative_target", _mode_or_first),
            pathway_name=("pathway_name", _mode_or_first),
            frac_ic50_within_range=(
                "ic50_within_range",
                lambda s: float(pd.Series(s).dropna().astype(bool).mean()) if s.notna().any() else np.nan,
            ),
        )
        .reset_index()
    )
    summary = summary.merge(meta, on="drug_name", how="left")
    summary["ic50_um_geomean"] = np.exp(summary["ln_ic50_mean"])

    # BH correction over drugs that actually have a test statistic.
    summary["z_q"] = np.nan
    testable = summary["z_p"].notna()
    if testable.any():
        summary.loc[testable, "z_q"] = multipletests(summary.loc[testable, "z_p"], method="fdr_bh")[1]

    summary["gbm_selective"] = (
        (summary["z_q"] < fdr) & (summary["z_mean"] <= z_effect) & (summary["n_cell_lines"] >= min_cell_lines)
    ).fillna(False)
    summary["potent"] = (summary["ln_ic50_mean"] < potency_threshold).fillna(False)

    ordered = [
        "drug_name",
        "drug_ids",
        "datasets",
        "putative_target",
        "pathway_name",
        "n_curves",
        "n_cell_lines",
        "ln_ic50_mean",
        "ln_ic50_median",
        "ln_ic50_sd",
        "ic50_um_geomean",
        "auc_mean",
        "frac_ic50_within_range",
        "z_mean",
        "z_sd",
        "z_t",
        "z_p",
        "z_q",
        "gbm_selective",
        "potent",
    ]
    summary = summary[ordered].sort_values("z_mean").reset_index(drop=True)
    logger.info(
        "Drug summary: %d drugs, %d GBM-selective at FDR %.2f, %d potent (ln IC50 < %.1f)",
        len(summary),
        int(summary["gbm_selective"].sum()),
        fdr,
        int(summary["potent"].sum()),
        potency_threshold,
    )
    return summary


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_processed(subset: pd.DataFrame, summary: pd.DataFrame) -> None:
    GBM_DOSE_RESPONSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(GBM_DOSE_RESPONSE_FILE, index=False, compression="gzip")
    summary.to_csv(DRUG_SUMMARY_FILE, index=False, float_format="%.6g")
    logger.info("Wrote %s and %s", GBM_DOSE_RESPONSE_FILE.name, DRUG_SUMMARY_FILE.name)


def load_gbm_dose_response(path: Path = GBM_DOSE_RESPONSE_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/build_gbm_dataset.py`.")
    return pd.read_csv(path)


def load_drug_summary(path: Path = DRUG_SUMMARY_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/build_gbm_dataset.py`.")
    return pd.read_csv(path)


def process_pipeline(
    paths: Iterable[Path] = (GDSC1_FILE, GDSC2_FILE), save: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw GDSC files -> (GBM curve subset, per-drug summary)."""
    raw = load_gdsc_raw(paths)
    subset = build_gbm_subset(raw)
    summary = summarize_drugs(subset)
    if save:
        save_processed(subset, summary)
    return subset, summary
