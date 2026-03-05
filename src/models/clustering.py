"""
Drug Clustering Analysis Module

Groups drugs by molecular similarity using unsupervised learning.

Algorithms:
1. KMeans - Centroid-based clustering (k=5)
2. DBSCAN - Density-based spatial clustering (eps=0.5, min_samples=3)
3. Hierarchical - Agglomerative clustering with Ward linkage

Dimensionality Reduction:
- PCA - Principal Component Analysis for feature reduction
- UMAP - Uniform Manifold Approximation and Projection for visualization

Metrics:
- Silhouette Score (cluster cohesion)
- Davies-Bouldin Index (cluster separation)
- Calinski-Harabasz Score (variance ratio)

Outputs:
- clustering_results.csv - Cluster assignments for each drug
- Metrics saved in results dictionary
- 2D/3D embeddings for visualization

Usage:
    clustering = DrugClusteringAnalyzer()
    results = clustering.analyze_all_methods(features_df, feature_cols)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import umap
from typing import Dict, Tuple, Optional, List
import logging

from ..config import (
    KMEANS_N_CLUSTERS, KMEANS_MAX_ITER, KMEANS_N_INIT, KMEANS_RANDOM_STATE,
    DBSCAN_EPS, DBSCAN_MIN_SAMPLES, DBSCAN_METRIC,
    HIERARCHICAL_N_CLUSTERS, HIERARCHICAL_LINKAGE,
    UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_N_COMPONENTS, UMAP_METRIC, UMAP_RANDOM_STATE,
    CLUSTERING_RESULTS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugClusteringAnalyzer:
    """Perform clustering analysis on drug features"""
    
    def __init__(self):
        """Initialize clustering analyzer"""
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.dbscan_model = None
        self.hierarchical_model = None
        self.scaled_features = None
    
    def preprocess_features(self, features_df: pd.DataFrame,
                           feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """
        Preprocess and scale features for clustering
        
        Args:
            features_df: DataFrame with features
            feature_cols: List of feature columns to use (None = use all numeric)
            
        Returns:
            Scaled feature array
        """
        if feature_cols is None:
            # Use all numeric columns
            feature_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Using {len(feature_cols)} features for clustering")
        
        # Extract features
        X = features_df[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.scaled_features = X_scaled
        
        return X_scaled
    
    def perform_kmeans(self, X: np.ndarray,
                       n_clusters: int = KMEANS_N_CLUSTERS,
                       random_state: int = KMEANS_RANDOM_STATE) -> Tuple[np.ndarray, KMeans]:
        """
        Perform K-Means clustering
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            random_state: Random seed
            
        Returns:
            Tuple of (cluster labels, model)
        """
        logger.info(f"Performing K-Means clustering with {n_clusters} clusters")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=KMEANS_MAX_ITER,
            n_init=KMEANS_N_INIT,
            random_state=random_state
        )
        
        labels = kmeans.fit_predict(X)
        
        self.kmeans_model = kmeans
        
        logger.info(f"K-Means clustering complete. Inertia: {kmeans.inertia_:.2f}")
        
        return labels, kmeans
    
    def perform_dbscan(self, X: np.ndarray,
                       eps: float = DBSCAN_EPS,
                       min_samples: int = DBSCAN_MIN_SAMPLES) -> Tuple[np.ndarray, DBSCAN]:
        """
        Perform DBSCAN clustering
        
        Args:
            X: Feature matrix
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
            
        Returns:
            Tuple of (cluster labels, model)
        """
        logger.info(f"Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})")
        
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=DBSCAN_METRIC
        )
        
        labels = dbscan.fit_predict(X)
        
        self.dbscan_model = dbscan
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"DBSCAN clustering complete. Clusters: {n_clusters}, Noise points: {n_noise}")
        
        return labels, dbscan
    
    def perform_hierarchical(self, X: np.ndarray,
                            n_clusters: int = HIERARCHICAL_N_CLUSTERS,
                            linkage: str = HIERARCHICAL_LINKAGE) -> Tuple[np.ndarray, AgglomerativeClustering]:
        """
        Perform Hierarchical clustering
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            
        Returns:
            Tuple of (cluster labels, model)
        """
        logger.info(f"Performing Hierarchical clustering with {n_clusters} clusters (linkage={linkage})")
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
        labels = hierarchical.fit_predict(X)
        
        self.hierarchical_model = hierarchical
        
        logger.info("Hierarchical clustering complete")
        
        return labels, hierarchical
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray,
                           method_name: str = "Clustering") -> Dict[str, float]:
        """
        Evaluate clustering quality
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            method_name: Name of clustering method
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Filter out noise points (label -1) for evaluation
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]
        
        n_clusters = len(set(labels_filtered))
        
        if n_clusters < 2 or len(labels_filtered) < 2:
            logger.warning(f"{method_name}: Not enough clusters or samples for evaluation")
            return {
                'n_clusters': n_clusters,
                'silhouette_score': -1,
                'davies_bouldin_score': -1,
                'calinski_harabasz_score': -1
            }
        
        metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_score(X_filtered, labels_filtered),
            'davies_bouldin_score': davies_bouldin_score(X_filtered, labels_filtered),
            'calinski_harabasz_score': calinski_harabasz_score(X_filtered, labels_filtered)
        }
        
        logger.info(f"{method_name} Metrics:")
        logger.info(f"  Number of clusters: {metrics['n_clusters']}")
        logger.info(f"  Silhouette Score: {metrics['silhouette_score']:.3f}")
        logger.info(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
        logger.info(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.3f}")
        
        return metrics
    
    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 11)) -> Tuple[int, Dict]:
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            X: Feature matrix
            k_range: Range of k values to test
            
        Returns:
            Tuple of (optimal k, results dict)
        """
        logger.info(f"Finding optimal K using range {k_range}")
        
        results = {
            'k_values': [],
            'inertias': [],
            'silhouette_scores': []
        }
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=10)
            labels = kmeans.fit_predict(X)
            
            results['k_values'].append(k)
            results['inertias'].append(kmeans.inertia_)
            results['silhouette_scores'].append(silhouette_score(X, labels))
        
        # Find k with best silhouette score
        optimal_k = results['k_values'][np.argmax(results['silhouette_scores'])]
        
        logger.info(f"Optimal K: {optimal_k} (Silhouette Score: {max(results['silhouette_scores']):.3f})")
        
        return optimal_k, results
    
    def reduce_dimensions_pca(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensions using PCA for visualization
        
        Args:
            X: Feature matrix
            n_components: Number of components
            
        Returns:
            Reduced feature matrix
        """
        pca = PCA(n_components=n_components, random_state=KMEANS_RANDOM_STATE)
        X_reduced = pca.fit_transform(X)
        
        logger.info(f"PCA: Explained variance ratio: {pca.explained_variance_ratio_}")
        
        return X_reduced
    
    def reduce_dimensions_umap(self, X: np.ndarray,
                              n_components: int = UMAP_N_COMPONENTS) -> np.ndarray:
        """
        Reduce dimensions using UMAP for visualization
        
        Args:
            X: Feature matrix
            n_components: Number of components
            
        Returns:
            Reduced feature matrix
        """
        logger.info("Performing UMAP dimensionality reduction")
        
        reducer = umap.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            n_components=n_components,
            metric=UMAP_METRIC,
            random_state=UMAP_RANDOM_STATE,
            n_jobs=1  # Explicitly set to match random_state behavior (reproducibility over parallelism)
        )
        
        X_reduced = reducer.fit_transform(X)
        
        logger.info("UMAP reduction complete")
        
        return X_reduced
    
    def analyze_all_methods(self, features_df: pd.DataFrame,
                           feature_cols: Optional[List[str]] = None) -> Dict:
        """
        Perform clustering with all methods and compare
        
        Args:
            features_df: DataFrame with features
            feature_cols: List of feature columns to use
            
        Returns:
            Dictionary with all results
        """
        logger.info("=" * 60)
        logger.info("Running clustering analysis with all methods")
        logger.info("=" * 60)
        
        # Preprocess features
        X = self.preprocess_features(features_df, feature_cols)
        
        results = {}
        
        # K-Means
        kmeans_labels, kmeans_model = self.perform_kmeans(X)
        results['kmeans'] = {
            'labels': kmeans_labels,
            'model': kmeans_model,
            'metrics': self.evaluate_clustering(X, kmeans_labels, "K-Means")
        }
        
        # DBSCAN
        dbscan_labels, dbscan_model = self.perform_dbscan(X)
        results['dbscan'] = {
            'labels': dbscan_labels,
            'model': dbscan_model,
            'metrics': self.evaluate_clustering(X, dbscan_labels, "DBSCAN")
        }
        
        # Hierarchical
        hierarchical_labels, hierarchical_model = self.perform_hierarchical(X)
        results['hierarchical'] = {
            'labels': hierarchical_labels,
            'model': hierarchical_model,
            'metrics': self.evaluate_clustering(X, hierarchical_labels, "Hierarchical")
        }
        
        # Add dimensionality reductions for visualization
        results['pca_2d'] = self.reduce_dimensions_pca(X, n_components=2)
        results['umap_2d'] = self.reduce_dimensions_umap(X, n_components=2)
        
        logger.info("=" * 60)
        logger.info("Clustering analysis complete")
        logger.info("=" * 60)
        
        return results
    
    def save_clustering_results(self, features_df: pd.DataFrame,
                               results: Dict,
                               filename: str = "clustering_results.csv"):
        """
        Save clustering results to file
        
        Args:
            features_df: Original features DataFrame
            results: Clustering results dictionary
            filename: Output filename
        """
        output_df = features_df.copy()
        
        # Add cluster labels
        if 'kmeans' in results:
            output_df['kmeans_cluster'] = results['kmeans']['labels']
        
        if 'dbscan' in results:
            output_df['dbscan_cluster'] = results['dbscan']['labels']
        
        if 'hierarchical' in results:
            output_df['hierarchical_cluster'] = results['hierarchical']['labels']
        
        # Add reduced dimensions
        if 'pca_2d' in results:
            output_df['pca_1'] = results['pca_2d'][:, 0]
            output_df['pca_2'] = results['pca_2d'][:, 1]
        
        if 'umap_2d' in results:
            output_df['umap_1'] = results['umap_2d'][:, 0]
            output_df['umap_2'] = results['umap_2d'][:, 1]
        
        # Save to file
        output_path = CLUSTERING_RESULTS_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        
        logger.info(f"Clustering results saved to {output_path}")


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Create dummy data
    X_dummy = np.random.randn(n_samples, n_features)
    features_df = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(n_features)])
    features_df['drug_name'] = [f'Drug_{i}' for i in range(n_samples)]
    
    # Perform clustering
    analyzer = DrugClusteringAnalyzer()
    results = analyzer.analyze_all_methods(features_df)
    
    print("\nClustering Results Summary:")
    for method in ['kmeans', 'dbscan', 'hierarchical']:
        print(f"\n{method.upper()}:")
        print(f"  Metrics: {results[method]['metrics']}")
