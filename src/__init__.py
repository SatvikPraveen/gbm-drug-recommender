"""
GBM Drug Analysis and Recommendation System

A comprehensive computational pipeline for precision oncology in Glioblastoma.

Modules:
- config: Central configuration (paths, hyperparameters, settings)
- data_processing: GDSC data loading and preprocessing
- feature_extraction: Molecular descriptor extraction from SMILES
- similarity: Three-method similarity analysis (Tanimoto, MCS, GCN)
- models: Machine learning (clustering, One-Class SVM, model comparison)
- pathway_analysis: Enrichr-based pathway enrichment
- combination_therapy: Drug pair synergy scoring
- drug_interactions: Safety analysis for combinations
- utils: Visualization and helper utilities

Version: 2.0.0 (Extended with combination therapy and ML comparison)
Author: GBM Research Team
"""

from . import config
from . import data_processing
from . import feature_extraction
from . import similarity
from . import models
from . import pathway_analysis
from . import utils

__version__ = "2.0.0"
__author__ = "GBM Research Team"

__all__ = [
    'config',
    'data_processing',
    'feature_extraction',
    'similarity',
    'models',
    'pathway_analysis',
    'utils'
]
