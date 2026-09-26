"""Tests for the GDSC -> GBM subset -> per-drug summary pipeline, on a synthetic GDSC-shaped frame."""

import numpy as np
import pandas as pd
import pytest

from gbm_drug import data_processing as dp


def _raw_frame() -> pd.DataFrame:
    """Two drugs x (3 GBM lines + 2 non-GBM lines), the first drug screened in both GDSC1 and GDSC2."""
    rows = []
    lines = [("U-87-MG", "GBM"), ("T98G", "GBM"), ("LN-229", "GBM"), ("A549", "LUAD"), ("MCF7", "BRCA")]
    for dataset, drug_id, name, target, pathway, ln_ic50, z in [
        ("GDSC1", 1, "Erlotinib", "EGFR", "EGFR signaling", -2.0, -1.5),
        ("GDSC2", 1001, "Erlotinib ", "EGFR", "EGFR signaling", -2.2, -1.7),  # trailing space on purpose
        ("GDSC2", 2, "Cisplatin", "DNA crosslinker", "DNA replication", 3.0, 0.8),
    ]:
        for i, (line, tcga) in enumerate(lines):
            rows.append(
                {
                    "DATASET": dataset,
                    "COSMIC_ID": 100 + i,
                    "CELL_LINE_NAME": line,
                    "SANGER_MODEL_ID": f"SIDM{i:05d}",
                    "TCGA_DESC": tcga,
                    "DRUG_ID": drug_id,
                    "DRUG_NAME": name,
                    "PUTATIVE_TARGET": target,
                    "PATHWAY_NAME": pathway,
                    "MIN_CONC": 0.001,
                    "MAX_CONC": 10.0,
                    "LN_IC50": ln_ic50 + 0.1 * i,
                    "AUC": 0.5,
                    "RMSE": 0.05,
                    "Z_SCORE": z + 0.05 * i,
                }
            )
    return pd.DataFrame(rows)


def test_build_gbm_subset_filters_by_tcga_label_and_standardises_columns():
    subset = dp.build_gbm_subset(_raw_frame())
    assert set(subset["tcga_desc"]) == {"GBM"}
    assert subset["cell_line"].nunique() == 3
    assert list(subset.columns) == dp.SUBSET_COLUMNS
    # Drug names are whitespace-normalised so GDSC1/GDSC2 spellings merge.
    assert set(subset["drug_name"]) == {"Erlotinib", "Cisplatin"}


def test_ic50_um_and_within_range_flag():
    subset = dp.build_gbm_subset(_raw_frame())
    np.testing.assert_allclose(subset["ic50_um"], np.exp(subset["ln_ic50"]))
    # Cisplatin ln IC50 ~3.0 => IC50 ~20 µM > MAX_CONC 10 µM => extrapolated.
    cis = subset[subset["drug_name"] == "Cisplatin"]
    assert not cis["ic50_within_range"].any()
    erl = subset[subset["drug_name"] == "Erlotinib"]
    assert erl["ic50_within_range"].all()


def test_build_gbm_subset_raises_when_label_absent():
    with pytest.raises(ValueError):
        dp.build_gbm_subset(_raw_frame(), tcga_label="NOT_A_LABEL")


def test_summary_averages_duplicate_screens_before_testing():
    subset = dp.build_gbm_subset(_raw_frame())
    summary = dp.summarize_drugs(subset, min_cell_lines=3)
    erl = summary.set_index("drug_name").loc["Erlotinib"]
    # 6 curves (3 lines x 2 datasets) but only 3 distinct cell lines feed the t-test.
    assert erl["n_curves"] == 6
    assert erl["n_cell_lines"] == 3
    assert erl["datasets"] == "GDSC1;GDSC2"
    assert erl["drug_ids"] == "1;1001"
    assert erl["putative_target"] == "EGFR"
    # Mean ln IC50 is the mean of per-line means: per-line = ((-2+0.1i)+(-2.2+0.1i))/2.
    expected = np.mean([(-2.0 + 0.1 * i + -2.2 + 0.1 * i) / 2 for i in range(3)])
    assert erl["ln_ic50_mean"] == pytest.approx(expected)


def test_selectivity_labels_follow_fdr_effect_and_min_lines():
    subset = dp.build_gbm_subset(_raw_frame())
    summary = dp.summarize_drugs(subset, fdr=0.5, min_cell_lines=3, z_effect=-0.5).set_index("drug_name")
    # Erlotinib: strongly negative z with tiny spread => significant, selective.
    assert summary.loc["Erlotinib", "z_mean"] < -0.5
    assert summary.loc["Erlotinib", "gbm_selective"]
    # Cisplatin: positive z => never selective regardless of p-value.
    assert not summary.loc["Cisplatin", "gbm_selective"]
    # Raising the minimum cell-line requirement above what we have removes the label.
    strict = dp.summarize_drugs(subset, fdr=0.5, min_cell_lines=10).set_index("drug_name")
    assert not strict["gbm_selective"].any()


def test_potency_label_uses_ln_ic50_threshold():
    subset = dp.build_gbm_subset(_raw_frame())
    summary = dp.summarize_drugs(subset, potency_threshold=0.0).set_index("drug_name")
    assert summary.loc["Erlotinib", "potent"]
    assert not summary.loc["Cisplatin", "potent"]


def test_q_values_are_bh_adjusted_and_bounded():
    subset = dp.build_gbm_subset(_raw_frame())
    summary = dp.summarize_drugs(subset)
    q = summary["z_q"].dropna()
    assert ((q >= 0) & (q <= 1)).all()
    assert (q >= summary.loc[q.index, "z_p"]).all()


def test_load_gdsc_raw_missing_file_has_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_gdsc.py"):
        dp.load_gdsc_raw([tmp_path / "nope.xlsx"])


def test_load_gdsc_raw_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing expected GDSC columns"):
        dp.load_gdsc_raw([bad])
