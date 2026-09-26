#!/usr/bin/env python
"""
Build the version-controlled GBM subset from the raw GDSC release files.

Writes:
    data/processed/gdsc_gbm_dose_response.csv.gz   one row per (dataset, drug_id, cell line)
    data/processed/gdsc_gbm_drug_summary.csv       one row per drug with potency/selectivity stats

Run scripts/download_gdsc.py first (it verifies checksums).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running as `python scripts/<name>.py` without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gbm_drug.data_processing import process_pipeline


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    subset, summary = process_pipeline()
    print(
        f"\nGBM subset:   {len(subset):>7} curves, {subset['cell_line'].nunique()} cell lines, {subset['drug_name'].nunique()} drugs"
    )
    print(
        f"Drug summary: {len(summary):>7} drugs, {int(summary['gbm_selective'].sum())} GBM-selective, {int(summary['potent'].sum())} potent"
    )
    print("\nMost GBM-selective drugs (lowest mean z-score):")
    cols = ["drug_name", "putative_target", "n_cell_lines", "z_mean", "z_q", "ic50_um_geomean"]
    print(summary[cols].head(15).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
