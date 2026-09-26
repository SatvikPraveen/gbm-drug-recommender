# Data

All inputs are public. Nothing in this directory was hand-edited except the
files explicitly marked *curated* below.

## Layout

```
data/
├── MANIFEST.json                 # GDSC release, URLs, SHA-256 checksums (the source of truth)
├── raw/                          # GDSC release files (gitignored; scripts/download_gdsc.py)
├── processed/
│   ├── gdsc_gbm_dose_response.csv.gz   # committed: every GBM curve from GDSC1+GDSC2
│   └── gdsc_gbm_drug_summary.csv       # committed: one row per drug with potency/selectivity stats
├── smiles/
│   ├── drug_smiles.csv           # committed: PubChem-resolved SMILES + RDKit annotations
│   ├── manual_overrides.csv      # curated: drug names PubChem cannot resolve by name
│   └── unresolved.txt            # drugs with no structure (biologics, unnamed screening compounds)
└── genesets/                     # Enrichr GMT libraries (gitignored; fetched on demand)
```

## Source: GDSC release 8.5 (27 Oct 2023)

Genomics of Drug Sensitivity in Cancer, Wellcome Sanger Institute / Massachusetts
General Hospital. Bulk download page: <https://www.cancerrxgene.org/downloads/bulk_download>.

| File | Rows | What it is |
|---|---|---|
| `GDSC1_fitted_dose_response_27Oct23.xlsx` | 333,161 | Fitted curves, screen 1 (2010–2015 chemistry, 72 h, resazurin/Syto60) |
| `GDSC2_fitted_dose_response_27Oct23.xlsx` | 242,036 | Fitted curves, screen 2 (2015 onward, CellTiter-Glo) |
| `screened_compounds_rel_8.5.csv` | 621 | Drug names, synonyms, targets, target pathways |
| `Cell_Lines_Details.xlsx` | 1,002 | Cell line annotation incl. TCGA label |

Reproduce the exact inputs with:

```bash
python scripts/download_gdsc.py      # fetches to data/raw/ and verifies SHA-256
python scripts/build_gbm_dataset.py  # regenerates data/processed/*
python scripts/fetch_smiles.py       # regenerates data/smiles/drug_smiles.csv from PubChem
```

If GDSC re-issues a release the checksums will fail and the download script
refuses to proceed; update `MANIFEST.json` deliberately and re-run.

## Cohort definition

GBM cell lines are those GDSC labels `TCGA_DESC == "GBM"`: **34 cell lines** across
the two screens. This replaces an earlier hard-coded name list (`U-87`, `U-251`,
`SNB-19`, …) that matched only 4 lines because GDSC spells them `U-87-MG`, `U251`,
`SNB75`.

Result: **20,329 curves, 542 distinct drug names**, 5–34 GBM lines per drug (median 33).

## Columns in `gdsc_gbm_dose_response.csv.gz`

| Column | Meaning |
|---|---|
| `dataset` | `GDSC1` or `GDSC2` |
| `drug_id`, `drug_name` | GDSC identifiers; names whitespace-normalised so both screens merge |
| `putative_target`, `pathway_name` | GDSC's own annotation |
| `cell_line`, `cosmic_id`, `sanger_model_id`, `tcga_desc` | Cell line identity |
| `min_conc_um`, `max_conc_um` | Tested concentration range (µM) |
| `ln_ic50` | Natural log of fitted IC50 (µM) |
| `ic50_um` | `exp(ln_ic50)` |
| `ic50_within_range` | `ln_ic50 <= ln(max_conc_um)`; **False means the IC50 is extrapolated beyond the tested range** |
| `auc` | Area under the fitted dose-response curve (1 = no effect) |
| `rmse` | Curve-fit residual |
| `z_score` | GDSC's per-drug z-score of ln IC50 across *all* screened cell lines |

## Columns in `gdsc_gbm_drug_summary.csv`

One row per drug name. Duplicate screens of the same drug on the same cell line
(GDSC1 vs GDSC2, or two `DRUG_ID`s) are averaged *before* any statistic so each
cell line counts once.

| Column | Meaning |
|---|---|
| `n_curves`, `n_cell_lines`, `datasets`, `drug_ids` | Provenance |
| `ln_ic50_mean`, `ln_ic50_median`, `ln_ic50_sd`, `ic50_um_geomean` | Potency in GBM lines (lower = more potent) |
| `auc_mean` | Mean AUC (lower = more sensitive) |
| `frac_ic50_within_range` | Share of curves whose IC50 lies inside the tested range |
| `z_mean`, `z_sd` | Mean and SD of GDSC z-scores over GBM lines. **Negative = GBM more sensitive than the pan-cancer panel** |
| `z_t`, `z_p`, `z_q` | One-sample t-test of z-scores against 0; `z_q` is Benjamini–Hochberg FDR over all drugs |
| `gbm_selective` | `z_q < 0.05` and `z_mean <= SELECTIVITY_Z_EFFECT` and `n_cell_lines >= 5` |
| `potent` | `ln_ic50_mean < 0` (geometric-mean IC50 below 1 µM) |

Thresholds are in `gbm_drug/config.py` and justified in `docs/METHODS.md`.

## Caveats you should know before modelling

- **Most IC50s are extrapolated.** The median drug has only 17 % of its GBM curves
  with an IC50 inside the tested range. `ln_ic50` beyond `max_conc_um` is a
  curve-fit extrapolation, not a measurement. `z_score` and `auc` are less
  affected, which is one reason selectivity is defined on z-scores.
- **Drug names are not structures.** `Bleomycin (10 uM)` and `Bleomycin (50 uM)`
  are the same molecule screened at two concentrations; several drugs have two
  `DRUG_ID`s. Structure-based models must group by InChIKey when splitting
  (see `gbm_drug/evaluation.py`), or the same molecule leaks across folds.
- **GDSC1 and GDSC2 used different viability assays.** Mixing them adds
  batch variance to `ln_ic50`; the z-score is computed within each screen.
- **Biologics have no SMILES.** Antibodies and other non-small-molecule entries
  are listed in `smiles/unresolved.txt` and excluded from structure-based analyses.

## Licence and citation

GDSC data are provided for academic, non-commercial use
(<https://www.cancerrxgene.org/legal>). The derived tables here are a filtered
view of that release and carry the same terms. If you use them, cite:

- Yang W. et al. *Genomics of Drug Sensitivity in Cancer (GDSC): a resource for
  therapeutic biomarker discovery in cancer cells.* Nucleic Acids Res 41, D955–D961 (2013).
- Iorio F. et al. *A Landscape of Pharmacogenomic Interactions in Cancer.*
  Cell 166, 740–754 (2016).

SMILES come from PubChem (Kim S. et al., Nucleic Acids Res 2023), public domain.
