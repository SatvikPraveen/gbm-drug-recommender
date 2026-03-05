"""
Configuration Module

Centralized configuration for the entire GBM Drug Analysis pipeline.

Contains:
- File paths (data, results, models, figures)
- Hyperparameters for all algorithms
  * Clustering: KMeans, DBSCAN, Hierarchical, UMAP
  * ML Models: SVM, KNN, Random Forest, Neural Network, XGBoost
  * Similarity: Tanimoto, MCS thresholds
- API endpoints (Enrichr, PubChem)
- Pathway databases
- Visualization settings
- IC50 effectiveness thresholds

Usage:
    from src.config import *
    # All constants and paths are now available

Modification:
    Edit values in this file to adjust pipeline behavior
    All modules import from this central configuration
"""

import os
from pathlib import Path
from typing import Dict, List

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# ==================== PATHS ====================

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SMILES_DATA_DIR = DATA_DIR / "smiles"

# Input files
GDSC1_FILE = RAW_DATA_DIR / "GDSC1.rds"
GDSC2_FILE = RAW_DATA_DIR / "GDSC2.rds"

# Processed data outputs
MERGED_DATA_FILE = PROCESSED_DATA_DIR / "merged_gdsc_data.csv"
FEATURES_FILE = PROCESSED_DATA_DIR / "molecular_features.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_gdsc_data.csv"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
SIMILARITY_RESULTS_DIR = RESULTS_DIR / "similarity"
CLUSTERING_RESULTS_DIR = RESULTS_DIR / "clustering"
MODEL_RESULTS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
PATHWAY_RESULTS_DIR = RESULTS_DIR / "pathways"

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, SMILES_DATA_DIR, RESULTS_DIR, 
                  SIMILARITY_RESULTS_DIR, CLUSTERING_RESULTS_DIR, 
                  MODEL_RESULTS_DIR, FIGURES_DIR, PATHWAY_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== GBM CELL LINES ====================
GBM_CELL_LINES = [
    'U-87',
    'U-251',
    'U-138',
    'SNB-19',
    'SF-268',
    'SF-295',
    'SF-539',
    'SNB-75',
    'T98G',
    'LN-229',
    'A172'
]

# ==================== DATA PROCESSING PARAMETERS ====================

# Missing value handling
IMPUTATION_STRATEGY = "mean"  # Options: 'mean', 'median', 'drop'
MISSING_THRESHOLD = 0.5  # Drop columns with >50% missing values

# Feature selection
IC50_THRESHOLD_EFFECTIVE = 10.0  # μM - drugs with IC50 < 10μM are considered effective
AUC_THRESHOLD_EFFECTIVE = 0.3  # Drugs with AUC < 0.3 are considered effective

# Z-score outlier detection
Z_SCORE_THRESHOLD = 3.0

# ==================== MOLECULAR DESCRIPTORS ====================

# RDKit molecular descriptors to extract
MOLECULAR_DESCRIPTORS = [
    'MolWt',           # Molecular weight
    'LogP',            # Lipophilicity
    'NumHDonors',      # Hydrogen bond donors
    'NumHAcceptors',   # Hydrogen bond acceptors
    'TPSA',            # Topological polar surface area
    'NumRotatableBonds',
    'NumAromaticRings',
    'FractionCSP3',
    'MolLogP',
    'MolMR'            # Molar refractivity
]

# Fingerprint parameters for similarity
FINGERPRINT_TYPE = "Morgan"  # Options: 'Morgan', 'MACCS', 'RDKit'
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048

# ==================== SIMILARITY ANALYSIS PARAMETERS ====================

# Tanimoto similarity threshold
TANIMOTO_THRESHOLD = 0.7  # Drugs with Tanimoto > 0.7 are considered similar

# MCS (Maximum Common Substructure) parameters
MCS_TIMEOUT = 2.0  # seconds
MCS_THRESHOLD = 0.6  # Similarity threshold for MCS

# GCN embedding parameters
GCN_HIDDEN_DIM = 64
GCN_OUTPUT_DIM = 128
GCN_NUM_LAYERS = 3
GCN_DROPOUT = 0.2
GCN_LEARNING_RATE = 0.001
GCN_EPOCHS = 100
GCN_BATCH_SIZE = 32

# Cosine similarity threshold for embeddings
COSINE_SIMILARITY_THRESHOLD = 0.8

# ==================== CLUSTERING PARAMETERS ====================

# K-Means
KMEANS_N_CLUSTERS = 5
KMEANS_MAX_ITER = 300
KMEANS_N_INIT = 10
KMEANS_RANDOM_STATE = 42

# DBSCAN
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5
DBSCAN_METRIC = 'euclidean'

# Hierarchical Clustering
HIERARCHICAL_N_CLUSTERS = 5
HIERARCHICAL_LINKAGE = 'ward'  # Options: 'ward', 'complete', 'average', 'single'

# UMAP for visualization
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2
UMAP_METRIC = 'euclidean'
UMAP_RANDOM_STATE = 42

# ==================== MACHINE LEARNING PARAMETERS ====================

# Train-test split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

# One-Class SVM
SVM_KERNEL = 'rbf'  # Options: 'rbf', 'linear', 'poly', 'sigmoid'
SVM_NU = 0.1  # Upper bound on fraction of outliers
SVM_GAMMA = 'scale'  # Options: 'scale', 'auto', or float

# Cross-validation
CV_FOLDS = 5
CV_SCORING = 'accuracy'

# Feature scaling
SCALER_TYPE = 'standard'  # Options: 'standard', 'minmax', 'robust'

# ==================== PATHWAY ANALYSIS PARAMETERS ====================

# Enrichr API settings
ENRICHR_URL = "https://maayanlab.cloud/Enrichr/addList"
ENRICHR_ENRICH_URL = "https://maayanlab.cloud/Enrichr/enrich"

# Gene set libraries to query
ENRICHR_LIBRARIES = [
    'KEGG_2021_Human',
    'WikiPathways_2021_Human',
    'Reactome_2022',
    'GO_Biological_Process_2021',
    'GO_Molecular_Function_2021',
    'BioPlanet_2019'
]

# Pathway significance threshold
PATHWAY_P_VALUE_THRESHOLD = 0.05
PATHWAY_ADJUSTED_P_VALUE_THRESHOLD = 0.05

# ==================== PYTORCH/MPS SETTINGS ====================

import torch

# Check for MPS (Metal Performance Shaders) availability on macOS
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ Using Metal Performance Shaders (MPS) for GPU acceleration")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✓ Using CUDA for GPU acceleration")
else:
    DEVICE = torch.device("cpu")
    print("ℹ Using CPU (consider enabling MPS on macOS for acceleration)")

# ==================== VISUALIZATION PARAMETERS ====================

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'  # Options: 'png', 'pdf', 'svg'
FIGURE_SIZE = (10, 8)

# Color palettes
COLORMAP_HEATMAP = 'viridis'
COLORMAP_CLUSTERS = 'tab10'
COLORMAP_SIMILARITY = 'RdYlBu_r'

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ==================== LOGGING ====================

LOG_LEVEL = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = PROJECT_ROOT / 'pipeline.log'

# ==================== OUTPUT PARAMETERS ====================

# Top N drugs to report
TOP_N_DRUGS = 20

# Similarity matrix export format
SIMILARITY_MATRIX_FORMAT = 'csv'  # Options: 'csv', 'hdf5', 'parquet'

# Model save format
MODEL_SAVE_FORMAT = 'pkl'  # Options: 'pkl', 'joblib', 'pt'

# ==================== HELPER FUNCTIONS ====================

def get_device_info() -> Dict:
    """Get information about the compute device being used"""
    return {
        'device': str(DEVICE),
        'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        'cuda_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU/MPS'
    }

def print_config_summary():
    """Print a summary of key configuration settings"""
    print("=" * 60)
    print("GBM DRUG ANALYSIS - CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"GBM Cell Lines: {len(GBM_CELL_LINES)}")
    print(f"Tanimoto Threshold: {TANIMOTO_THRESHOLD}")
    print(f"IC50 Threshold: {IC50_THRESHOLD_EFFECTIVE} μM")
    print(f"Clustering Methods: KMeans, DBSCAN, Hierarchical")
    print(f"ML Model: One-Class SVM (kernel={SVM_KERNEL}, nu={SVM_NU})")
    print("=" * 60)

if __name__ == "__main__":
    print_config_summary()
