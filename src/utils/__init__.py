"""
Utilities Package

Helper modules for visualization and data handling.

Tools:
- VisualizationTools: Publication-quality plots for all pipeline stages
  * Similarity heatmaps and distributions
  * Clustering visualizations (2D/3D UMAP)
  * Top drug rankings
  * Pathway enrichment bar charts
  * Model performance metrics

All visualizations saved to results/figures/ in PNG format.
Interactive Plotly charts available in dashboard.

Usage:
    from src.utils import VisualizationTools
    viz = VisualizationTools()
    viz.plot_similarity_heatmap(matrix, title, save_name)
"""

from .visualization import VisualizationTools

__all__ = ['VisualizationTools']
