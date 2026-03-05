"""
Visualization Utilities Module

Generate publication-quality plots for all pipeline stages.

Plot Types:
- Similarity Heatmaps - Drug-drug similarity matrices (Tanimoto, MCS, GCN)
- Similarity Distributions - Histograms of pairwise similarities
- Clustering Plots - 2D/3D UMAP projections with cluster colors
- Top Drugs Bar Charts - Ranked predictions with decision scores
- Pathway Enrichment - Horizontal bar plots of -log10(p-values)
- Model Performance - ROC curves, confusion matrices

Styling:
- Consistent color schemes (viridis, plasma)
- High-resolution output (300 DPI)
- Publication-ready fonts and labels
- Interactive Plotly charts for dashboard

Outputs:
- All figures saved to results/figures/
- PNG format for reports, HTML for interactive viewing

Usage:
    viz = VisualizationTools()
    viz.plot_similarity_heatmap(matrix, title, save_name)
    viz.plot_clustering_2d(embeddings, labels, title, save_name)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Tuple, Dict
import logging

from ..config import (
    FIGURE_DPI, FIGURE_FORMAT, FIGURE_SIZE, COLORMAP_HEATMAP,
    COLORMAP_CLUSTERS, COLORMAP_SIMILARITY, PLOT_STYLE, FIGURES_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
try:
    plt.style.use(PLOT_STYLE)
except:
    plt.style.use('default')


class VisualizationTools:
    """Tools for creating visualizations"""
    
    def __init__(self, output_dir: str = FIGURES_DIR):
        """
        Initialize visualization tools
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_similarity_heatmap(self, similarity_matrix: pd.DataFrame,
                               title: str = "Drug Similarity Matrix",
                               cmap: str = COLORMAP_SIMILARITY,
                               figsize: Tuple[int, int] = (12, 10),
                               save_name: Optional[str] = None):
        """
        Plot similarity matrix as heatmap
        
        Args:
            similarity_matrix: Similarity matrix DataFrame
            title: Plot title
            cmap: Colormap
            figsize: Figure size
            save_name: Filename to save (None = don't save)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            similarity_matrix,
            cmap=cmap,
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Similarity Score'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Drug', fontsize=12)
        ax.set_ylabel('Drug', fontsize=12)
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved figure: {filepath}")
        
        plt.close()  # Changed from plt.show() to avoid blocking
        return fig
    
    def plot_clustering_2d(self, X_2d: np.ndarray, labels: np.ndarray,
                          drug_names: Optional[List[str]] = None,
                          title: str = "Drug Clustering",
                          method: str = "PCA",
                          save_name: Optional[str] = None):
        """
        Plot 2D clustering results
        
        Args:
            X_2d: 2D coordinates
            labels: Cluster labels
            drug_names: Drug names for annotations
            title: Plot title
            method: Dimensionality reduction method name
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Get unique labels (excluding noise label -1)
        unique_labels = set(labels)
        colors = plt.cm.get_cmap(COLORMAP_CLUSTERS)(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Noise points in black
                col = 'black'
                label = 'Noise'
            else:
                label = f'Cluster {k}'
            
            class_mask = labels == k
            
            ax.scatter(
                X_2d[class_mask, 0],
                X_2d[class_mask, 1],
                c=[col],
                label=label,
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel(f'{method} Component 1', fontsize=12)
        ax.set_ylabel(f'{method} Component 2', fontsize=12)
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved figure: {filepath}")
        
        plt.close()  # Changed from plt.show() to avoid blocking
        return fig
    
    def plot_interactive_clustering(self, X_2d: np.ndarray, labels: np.ndarray,
                                   drug_names: List[str],
                                   title: str = "Interactive Drug Clustering",
                                   method: str = "UMAP",
                                   save_name: Optional[str] = None):
        """
        Create interactive Plotly clustering visualization
        
        Args:
            X_2d: 2D coordinates
            labels: Cluster labels
            drug_names: Drug names
            title: Plot title
            method: Dimensionality reduction method
            save_name: Filename to save (HTML)
        """
        df_plot = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            'cluster': [f'Cluster {l}' if l != -1 else 'Noise' for l in labels],
            'drug': drug_names
        })
        
        fig = px.scatter(
            df_plot,
            x='x',
            y='y',
            color='cluster',
            hover_data=['drug'],
            title=title,
            labels={'x': f'{method} 1', 'y': f'{method} 2'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(
            font=dict(size=12),
            title_font=dict(size=16),
            hovermode='closest',
            width=900,
            height=700
        )
        
        if save_name:
            filepath = self.output_dir / save_name
            fig.write_html(filepath)
            logger.info(f"Saved interactive figure: {filepath}")
        
        fig.show()
        return fig
    
    def plot_drug_response_distribution(self, data: pd.DataFrame,
                                       metric: str = 'ic50',
                                       title: Optional[str] = None,
                                       save_name: Optional[str] = None):
        """
        Plot distribution of drug response metrics
        
        Args:
            data: DataFrame with drug response data
            metric: Metric column name ('ic50', 'auc', etc.)
            title: Plot title
            save_name: Filename to save
        """
        if metric not in data.columns:
            logger.warning(f"Metric '{metric}' not found in data")
            return None
        
        if title is None:
            title = f"Distribution of {metric.upper()}"
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(data[metric].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel(metric.upper(), fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'{metric.upper()} Distribution', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(data[metric].dropna(), vert=True)
        axes[1].set_ylabel(metric.upper(), fontsize=12)
        axes[1].set_title(f'{metric.upper()} Box Plot', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved figure: {filepath}")
        
        plt.close()  # Changed from plt.show() to avoid blocking
        return fig
    
    def plot_top_drugs(self, drugs_df: pd.DataFrame,
                      score_col: str,
                      drug_name_col: str = 'drug_name',
                      top_n: int = 20,
                      title: str = "Top Drugs by Score",
                      save_name: Optional[str] = None):
        """
        Plot top drugs by score
        
        Args:
            drugs_df: DataFrame with drug scores
            score_col: Score column name
            drug_name_col: Drug name column
            top_n: Number of top drugs to show
            title: Plot title
            save_name: Filename to save
        """
        # Sort and get top N
        top_drugs = drugs_df.nlargest(top_n, score_col)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        # Horizontal bar plot
        y_pos = np.arange(len(top_drugs))
        ax.barh(y_pos, top_drugs[score_col], align='center', alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_drugs[drug_name_col])
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title(title, fontsize=16, pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved figure: {filepath}")
        
        plt.close()  # Changed from plt.show() to avoid blocking
        return fig
    
    def plot_similarity_distribution(self, similarity_matrix: pd.DataFrame,
                                    title: str = "Similarity Score Distribution",
                                    save_name: Optional[str] = None):
        """
        Plot distribution of similarity scores
        
        Args:
            similarity_matrix: Similarity matrix
            title: Plot title
            save_name: Filename to save
        """
        # Get upper triangle values (excluding diagonal)
        upper_triangle = np.triu(similarity_matrix.values, k=1)
        similarities = upper_triangle[upper_triangle > 0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(similarities.mean(), color='red', linestyle='--', 
                   label=f'Mean: {similarities.mean():.3f}')
        ax.axvline(np.median(similarities), color='blue', linestyle='--',
                   label=f'Median: {np.median(similarities):.3f}')
        
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=16, pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved figure: {filepath}")
        
        plt.close()  # Changed from plt.show() to avoid blocking
        return fig
    
    def plot_pathway_enrichment(self, pathway_df: pd.DataFrame,
                               top_n: int = 15,
                               title: str = "Pathway Enrichment",
                               save_name: Optional[str] = None):
        """
        Plot pathway enrichment results
        
        Args:
            pathway_df: DataFrame with pathway enrichment results
            top_n: Number of pathways to show
            title: Plot title
            save_name:Filename to save
        """
        if 'Term' not in pathway_df.columns or 'Adjusted P-value' not in pathway_df.columns:
            logger.warning("Required columns not found in pathway DataFrame")
            return None
        
        # Get top pathways
        top_pathways = pathway_df.nsmallest(top_n, 'Adjusted P-value')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.4)))
        
        # Horizontal bar plot with -log10(p-value)
        y_pos = np.arange(len(top_pathways))
        scores = -np.log10(top_pathways['Adjusted P-value'])
        
        bars = ax.barh(y_pos, scores, align='center', alpha=0.7, edgecolor='black')
        
        # Color bars by score
        colors = plt.cm.get_cmap(COLORMAP_HEATMAP)(scores / scores.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([term[:60] + '...' if len(term) > 60 else term 
                           for term in top_pathways['Term']], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('-log10(Adjusted P-value)', fontsize=12)
        ax.set_title(title, fontsize=16, pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add significance line
        ax.axvline(-np.log10(0.05), color='red', linestyle='--', 
                   alpha=0.5, label='p=0.05')
        ax.legend()
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved figure: {filepath}")
        
        plt.close()  # Changed from plt.show() to avoid blocking
        return fig
    
    def plot_comparison_matrix(self, methods_dict: Dict[str, pd.DataFrame],
                              title: str = "Similarity Method Comparison",
                              save_name: Optional[str] = None):
        """
        Compare multiple similarity matrices
        
        Args:
            methods_dict: Dictionary mapping method names to similarity matrices
            title: Plot title
            save_name: Filename to save
        """
        n_methods = len(methods_dict)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        for ax, (method_name, sim_matrix) in zip(axes, methods_dict.items()):
            sns.heatmap(
                sim_matrix,
                cmap=COLORMAP_SIMILARITY,
                vmin=0,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Similarity'},
                ax=ax,
                xticklabels=False,
                yticklabels=False
            )
            ax.set_title(method_name, fontsize=14)
        
        fig.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
            logger.info(f"Saved figure: {filepath}")
        
        plt.close()  # Changed from plt.show() to avoid blocking
        return fig


if __name__ == "__main__":
    # Example usage
    viz = VisualizationTools()
    
    # Create dummy similarity matrix
    np.random.seed(42)
    n = 10
    drugs = [f'Drug_{i}' for i in range(n)]
    sim_matrix = np.random.rand(n, n)
    sim_matrix = (sim_matrix + sim_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(sim_matrix, 1.0)
    
    sim_df = pd.DataFrame(sim_matrix, index=drugs, columns=drugs)
    
    print("Creating example visualizations...")
    viz.plot_similarity_heatmap(sim_df, title="Example Similarity Matrix")
    viz.plot_similarity_distribution(sim_df, title="Example Similarity Distribution")
