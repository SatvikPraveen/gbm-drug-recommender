# Changelog

## 3.0.0 — 2026-09-26

A rewrite aimed at making the project's claims reproducible and its
evaluation sound. Results from earlier versions should not be cited.

### Data
- Reproducible acquisition of GDSC release 8.5 with SHA-256 checksums
  (`data/MANIFEST.json`, `scripts/download_gdsc.py`).
- GBM cohort defined by GDSC's `TCGA_DESC == "GBM"`: 34 cell lines, 542 drugs,
  20,329 curves (previously 4 lines / 20 drugs / 76 rows, because a hard-coded
  name list did not match GDSC's spelling).
- Per-drug response table with GBM-selectivity statistics (GDSC z-score,
  one-sample t-test, BH-FDR) and provenance columns; committed.
- SMILES for 429 drugs from PubChem with an audit trail and curated overrides
  (`data/smiles/`); bevacizumab (an antibody) no longer carries the SMILES of
  isobutylbenzene.

### Evaluation
- Leakage-free benchmark: one row per molecule, folds grouped by InChIKey
  connectivity or Bemis–Murcko scaffold, repeated seeds, bootstrap CIs, dummy
  baselines, y-scrambling null model, out-of-fold predictions. The earlier
  "0.631 CV accuracy" came from splitting drug×cell-line rows at random with
  per-drug features.
- One-Class SVM scores are now out-of-fold; the earlier "13 promising
  candidates" were the 13 training drugs.
- GDSC pathway-annotation models added as a comparator to structure.
- Nested, molecule-grouped cross-validation selects tabular hyper-parameters
  inside each outer training fold; choices are recorded per fold.

### Analyses
- MCS similarity via `rdFMCS` (the earlier matrix was the identity).
- "GCN similarity" replaced by embeddings from a GNN actually trained on GBM
  response (the earlier network was untrained).
- Combination scoring uses GDSC target/pathway annotations (the earlier
  version received none, ranking two EGFR inhibitors as most complementary).
- Offline pathway enrichment with the screened library as background.
- Interaction screen reports neutral alert levels; the CYP table that
  contained none of the screened drugs is gone.

### Engineering
- Installable package `gbm_drug` with `pyproject.toml`, ruff, pytest (80+
  tests), GitHub Actions CI, pinned `requirements.lock`, run metadata
  (`results/metadata.json`) and an auto-generated `results/RESULTS.md`.

## 2.1.0 and earlier

See git history. These versions ran on a 20-drug sample and their reported
metrics are superseded.
