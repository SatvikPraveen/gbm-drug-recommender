"""
Molecular similarity: fingerprint Tanimoto, maximum common substructure, and
task-aware GNN embeddings, plus the Mantel test for comparing them.
"""

from .compare import mantel_test, summarize_matrix, upper_triangle
from .gnn_similarity import gnn_embeddings, gnn_similarity_matrix
from .mcs_similarity import mcs_matrix, mcs_similarity
from .tanimoto import pairs_above, tanimoto_matrix

__all__ = [
    "gnn_embeddings",
    "gnn_similarity_matrix",
    "mantel_test",
    "mcs_matrix",
    "mcs_similarity",
    "pairs_above",
    "summarize_matrix",
    "tanimoto_matrix",
    "upper_triangle",
]
