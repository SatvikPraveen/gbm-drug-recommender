"""
Similarity Analysis Package

Multi-method molecular similarity computation.

Analyzers:
- TanimotoSimilarityAnalyzer: Fingerprint-based similarity (Morgan/MACCS)
- MCSimilarityAnalyzer: Maximum Common Substructure matching
- GCNSimilarityAnalyzer: Graph Neural Network learned embeddings
- MolecularGCN: 3-layer GCN architecture for molecular graphs

Each method provides complementary views of drug relationships:
- Tanimoto: Fast, interpretable, structure-based
- MCS: Scaffold-focused, identifies conserved fragments
- GCN: Learned representations, captures complex patterns

Usage:
    from src.similarity import TanimotoSimilarityAnalyzer
    analyzer = TanimotoSimilarityAnalyzer()
    matrix = analyzer.build_similarity_matrix(smiles_dict)
"""

from .tanimoto import TanimotoSimilarityAnalyzer
from .mcs_similarity import MCSimilarityAnalyzer
from .gcn_similarity import GCNSimilarityAnalyzer, MolecularGCN

__all__ = [
    'TanimotoSimilarityAnalyzer',
    'MCSimilarityAnalyzer',
    'GCNSimilarityAnalyzer',
    'MolecularGCN'
]
