"""
Central configuration for the GBM drug-recommender pipeline.

This module holds only constants and a few small helpers. Importing it has no
side effects: it does not create directories, print, or touch PyTorch. Callers
that need directories call :func:`ensure_directories`; callers that need a
compute device call :func:`get_device`.

Every threshold below is a modelling choice, not a fact of nature. The rationale
for each is recorded in ``docs/METHODS.md``; change values here and re-run the
pipeline rather than editing downstream modules.
"""

from __future__ import annotations

from pathlib import Path

# ==================== PATHS ====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SMILES_DATA_DIR = DATA_DIR / "smiles"

# GDSC inputs. The download script (``scripts/download_gdsc.py``) writes these;
# ``data/MANIFEST.json`` records the release, URLs and SHA-256 checksums.
GDSC_RELEASE = "8.5"
GDSC1_FILE = RAW_DATA_DIR / "GDSC1_fitted_dose_response_27Oct23.xlsx"
GDSC2_FILE = RAW_DATA_DIR / "GDSC2_fitted_dose_response_27Oct23.xlsx"
GDSC_COMPOUNDS_FILE = RAW_DATA_DIR / "screened_compounds_rel_8.5.csv"
GDSC_CELL_LINES_FILE = RAW_DATA_DIR / "Cell_Lines_Details.xlsx"

# Derived, version-controlled data products.
GBM_DOSE_RESPONSE_FILE = PROCESSED_DATA_DIR / "gdsc_gbm_dose_response.csv.gz"
DRUG_SUMMARY_FILE = PROCESSED_DATA_DIR / "gdsc_gbm_drug_summary.csv"
SMILES_FILE = SMILES_DATA_DIR / "drug_smiles.csv"
SMILES_OVERRIDES_FILE = SMILES_DATA_DIR / "manual_overrides.csv"
FEATURES_FILE = PROCESSED_DATA_DIR / "molecular_features.csv"

# Legacy names kept so older notebooks keep importing.
MERGED_DATA_FILE = PROCESSED_DATA_DIR / "merged_gdsc_data.csv"
CLEANED_DATA_FILE = GBM_DOSE_RESPONSE_FILE

RESULTS_DIR = PROJECT_ROOT / "results"
SIMILARITY_RESULTS_DIR = RESULTS_DIR / "similarity"
CLUSTERING_RESULTS_DIR = RESULTS_DIR / "clustering"
MODEL_RESULTS_DIR = RESULTS_DIR / "models"
BENCHMARK_RESULTS_DIR = RESULTS_DIR / "benchmark"
FIGURES_DIR = RESULTS_DIR / "figures"
PATHWAY_RESULTS_DIR = RESULTS_DIR / "pathways"
COMBINATION_RESULTS_DIR = RESULTS_DIR / "combination_therapy"
INTERACTION_RESULTS_DIR = RESULTS_DIR / "interactions"

LOG_FILE = PROJECT_ROOT / "pipeline.log"

ALL_OUTPUT_DIRS = (
    PROCESSED_DATA_DIR,
    SMILES_DATA_DIR,
    RESULTS_DIR,
    SIMILARITY_RESULTS_DIR,
    CLUSTERING_RESULTS_DIR,
    MODEL_RESULTS_DIR,
    BENCHMARK_RESULTS_DIR,
    FIGURES_DIR,
    PATHWAY_RESULTS_DIR,
    COMBINATION_RESULTS_DIR,
    INTERACTION_RESULTS_DIR,
)


def ensure_directories() -> None:
    """Create every output directory the pipeline writes to."""
    for directory in ALL_OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# ==================== COHORT DEFINITION ====================

# GBM cell lines are selected by GDSC's own TCGA classification rather than by a
# hard-coded name list: GDSC spells lines as "U-87-MG", "U251", "SNB75", which a
# name list written from memory silently misses.
GBM_TCGA_LABEL = "GBM"

# ==================== RESPONSE LABELS ====================

# A drug is "GBM-selective" when GBM lines are significantly more sensitive to it
# than the pan-cancer panel. GDSC's Z_SCORE is already standardised per drug across
# all screened lines, so a one-sample t-test of GBM z-scores against 0 with
# Benjamini-Hochberg correction is the natural test. See docs/METHODS.md.
SELECTIVITY_FDR = 0.05
SELECTIVITY_MIN_CELL_LINES = 5  # need enough GBM lines for the t-test to mean anything
SELECTIVITY_Z_EFFECT = -0.5  # and a minimum effect size, in pan-cancer SDs

# Potency label used for the secondary classification task: mean ln(IC50) below
# ln(1 µM). Kept as a sensitivity analysis, not the primary endpoint.
POTENCY_LN_IC50_THRESHOLD = 0.0

# ==================== MOLECULAR DESCRIPTORS ====================

MOLECULAR_DESCRIPTORS = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "NumRotatableBonds",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "HeavyAtomCount",
    "RingCount",
    "NumHeteroatoms",
    "qed",
]

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048

# Compounds heavier than this are not small molecules (antibodies, peptides) and
# are excluded from structure-based analyses.
MAX_SMALL_MOLECULE_MW = 1500.0

# ==================== SIMILARITY ====================

TANIMOTO_THRESHOLD = 0.7
MCS_TIMEOUT_SECONDS = 1.0
MCS_MAX_DRUGS = 120  # MCS is O(n^2) and slow; restrict to the candidate set

# ==================== END-TO-END GNN ====================

GNN_HIDDEN_CHANNELS = 128
GNN_NUM_GNN_LAYERS = 3
GNN_NUM_MLP_LAYERS = 2
GNN_DROPOUT = 0.2
GNN_TYPE = "gcn"  # 'gcn' | 'gat'
GNN_POOLING = "mean"  # 'mean' | 'max' | 'add'
GNN_LEARNING_RATE = 1e-3
GNN_WEIGHT_DECAY = 1e-5
GNN_BATCH_SIZE = 32
GNN_EPOCHS = 200
GNN_EARLY_STOPPING_PATIENCE = 20
GNN_VALIDATION_SPLIT = 0.15

# ==================== CLUSTERING ====================

KMEANS_N_CLUSTERS = 6
KMEANS_N_INIT = 10
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 4
HIERARCHICAL_N_CLUSTERS = 6
HIERARCHICAL_LINKAGE = "ward"
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2

# ==================== EVALUATION PROTOCOL ====================

RANDOM_STATE = 42
CV_FOLDS = 5
CV_REPEATS = 5
BOOTSTRAP_SAMPLES = 1000
Y_SCRAMBLE_ROUNDS = 20

# ==================== ONE-CLASS SVM (exploratory) ====================

SVM_KERNEL = "rbf"
SVM_NU = 0.1
SVM_GAMMA = "scale"

# ==================== PATHWAY ENRICHMENT ====================

GENESET_DIR = DATA_DIR / "genesets"
GENESET_LIBRARIES = [
    "KEGG_2021_Human",
    "Reactome_2022",
    "WikiPathway_2023_Human",
]
ENRICHR_LIBRARY_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={library}"
PATHWAY_FDR = 0.05
PATHWAY_MIN_OVERLAP = 2

GBM_PATHWAY_KEYWORDS = [
    "EGFR",
    "VEGF",
    "PDGF",
    "PI3K",
    "AKT",
    "mTOR",
    "MAPK",
    "RAS",
    "glioma",
    "p53",
    "cell cycle",
    "DNA repair",
    "apoptosis",
]

# ==================== COMBINATION SCORING ====================

COMBINATION_WEIGHTS = {
    "target_diversity": 0.35,
    "pathway_complementarity": 0.25,
    "potency": 0.30,
    "structural_novelty": 0.10,
}
COMBINATION_MAX_TANIMOTO = 0.7  # pairs above this are near-duplicates, not combinations

# ==================== VISUALISATION ====================

FIGURE_DPI = 200
FIGURE_FORMAT = "png"
COLORMAP_SIMILARITY = "viridis"
COLORMAP_CLUSTERS = "tab10"
PLOT_STYLE = "seaborn-v0_8-whitegrid"

# ==================== LOGGING ====================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== OUTPUT ====================

TOP_N_DRUGS = 30


# ==================== HELPERS ====================


def get_device(preference: str = "auto") -> str:
    """Return the torch device string to use, without importing torch at module import."""
    if preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def config_summary() -> dict:
    """Machine-readable snapshot of the settings that affect results."""
    return {
        "gdsc_release": GDSC_RELEASE,
        "gbm_tcga_label": GBM_TCGA_LABEL,
        "selectivity_fdr": SELECTIVITY_FDR,
        "selectivity_min_cell_lines": SELECTIVITY_MIN_CELL_LINES,
        "selectivity_z_effect": SELECTIVITY_Z_EFFECT,
        "potency_ln_ic50_threshold": POTENCY_LN_IC50_THRESHOLD,
        "descriptors": list(MOLECULAR_DESCRIPTORS),
        "fingerprint": {"type": "Morgan", "radius": FINGERPRINT_RADIUS, "bits": FINGERPRINT_BITS},
        "cv": {"folds": CV_FOLDS, "repeats": CV_REPEATS, "random_state": RANDOM_STATE},
        "gnn": {
            "type": GNN_TYPE,
            "hidden": GNN_HIDDEN_CHANNELS,
            "layers": GNN_NUM_GNN_LAYERS,
            "dropout": GNN_DROPOUT,
            "lr": GNN_LEARNING_RATE,
            "epochs": GNN_EPOCHS,
        },
        "combination_weights": dict(COMBINATION_WEIGHTS),
    }


# ==================== LEGACY ALIASES ====================
# Names still imported by modules that have not yet been migrated to the new
# data model. Each is deleted in the commit that rewrites its consumer.

GBM_CELL_LINES: list[str] = []  # superseded by GBM_TCGA_LABEL
IMPUTATION_STRATEGY = "drop"
MISSING_THRESHOLD = 0.5
Z_SCORE_THRESHOLD = 3.0
IC50_THRESHOLD_EFFECTIVE = 10.0
AUC_THRESHOLD_EFFECTIVE = 0.3
FINGERPRINT_TYPE = "Morgan"
MCS_TIMEOUT = MCS_TIMEOUT_SECONDS
MCS_THRESHOLD = 0.6
COSINE_SIMILARITY_THRESHOLD = 0.8
GCN_HIDDEN_DIM = 64
GCN_OUTPUT_DIM = 128
GCN_NUM_LAYERS = 3
GCN_DROPOUT = 0.2
GCN_LEARNING_RATE = 1e-3
GCN_EPOCHS = 100
GCN_BATCH_SIZE = 32
CV_SCORING = "accuracy"
SCALER_TYPE = "standard"
TEST_SIZE = 0.2
ENRICHR_URL = "https://maayanlab.cloud/Enrichr/addList"
ENRICHR_ENRICH_URL = "https://maayanlab.cloud/Enrichr/enrich"
ENRICHR_LIBRARIES = list(GENESET_LIBRARIES)
PATHWAY_P_VALUE_THRESHOLD = 0.05
PATHWAY_ADJUSTED_P_VALUE_THRESHOLD = PATHWAY_FDR
FIGURE_SIZE = (10, 8)
COLORMAP_HEATMAP = "viridis"
KMEANS_MAX_ITER = 300
KMEANS_RANDOM_STATE = RANDOM_STATE
DBSCAN_METRIC = "euclidean"
UMAP_METRIC = "euclidean"
UMAP_RANDOM_STATE = RANDOM_STATE
