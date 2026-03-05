# Data Directory

This directory contains the GDSC drug sensitivity datasets used for GBM drug analysis.

## Directory Structure

```
data/
├── raw/              # Original datasets (place GDSC files here)
├── processed/        # Cleaned and merged datasets
└── smiles/           # SMILES string mappings
```

## Required Files

Place the following files in `data/raw/`:

1. **GDSC1.rds** - GDSC1 drug sensitivity dataset
2. **GDSC2.rds** - GDSC2 drug sensitivity dataset

## Data Source

The GDSC (Genomics of Drug Sensitivity in Cancer) datasets are available from:
- Website: https://www.cancerrxgene.org/
- Data portal: https://www.cancerrxgene.org/downloads

## Dataset Description

### GDSC1
- Focuses on standard chemotherapy agents
- Contains drug response data across hundreds of cancer cell lines
- Includes IC50, AUC, and Z-score measurements

### GDSC2
- Includes targeted therapies and newer compounds
- Complementary to GDSC1 with overlapping cell lines
- Similar response metrics (IC50, AUC, Z-score)

## Important Columns

After processing, the merged dataset includes:

- `cell_line`: Cell line name (e.g., U-87, U-251 for GBM)
- `drug_name`: Name of the drug
- `ic50`: Half-maximal inhibitory concentration (μM)
- `auc`: Area under the dose-response curve
- `z_score`: Standardized sensitivity score
- `tissue`: Tissue type (filtered for GBM/brain)
- `target`: Putative drug target
- `pathway`: Associated biological pathway

## GBM Cell Lines

The following GBM cell lines are filtered and analyzed:

- U-87
- U-251
- U-138
- SNB-19
- SF-268
- SF-295
- SF-539
- SNB-75
- T98G
- LN-229
- A172

## Data Processing

The raw data undergoes the following processing steps:

1. Load from RDS format using `pyreadr`
2. Standardize column names
3. Merge GDSC1 and GDSC2
4. Filter for GBM cell lines
5. Convert LN_IC50 to IC50 (μM)
6. Handle missing values (mean imputation)
7. Remove outliers (Z-score > 3)
8. Add efficacy labels (IC50 < 10 μM)

Processed data is saved to `data/processed/cleaned_gdsc_data.csv`

## Usage Terms

Please ensure compliance with GDSC data usage policies:
- Data is for research purposes only
- Cite GDSC publications in any research outputs
- Do not redistribute raw data

## References

Yang, W., Soares, J., Greninger, P. et al. Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. Nucleic Acids Res. 41, D955–D961 (2013).
