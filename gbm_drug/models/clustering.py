"""
Unsupervised structure of the drug set: clustering and 2-D embeddings.

Clustering is descriptive, not predictive: it shows which drugs are neighbours
in descriptor space and whether GBM-selective drugs concentrate anywhere.
K is chosen by silhouette over a range rather than fixed; DBSCAN and Ward
linkage are run as alternative views. Silhouette, Davies–Bouldin and
Calinski–Harabasz are reported for each.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import (
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    HIERARCHICAL_LINKAGE,
    KMEANS_N_INIT,
    RANDOM_STATE,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
)

logger = logging.getLogger(__name__)


def cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2 or mask.sum() < 3:
        return {
            "n_clusters": n_clusters,
            "n_noise": int((~mask).sum()),
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }
    return {
        "n_clusters": n_clusters,
        "n_noise": int((~mask).sum()),
        "silhouette": float(silhouette_score(X[mask], labels[mask])),
        "davies_bouldin": float(davies_bouldin_score(X[mask], labels[mask])),
        "calinski_harabasz": float(calinski_harabasz_score(X[mask], labels[mask])),
    }


def choose_k(
    X: np.ndarray, k_range=range(2, 11), random_state: int = RANDOM_STATE
) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=random_state).fit_predict(X)
        rows.append({"k": k, "silhouette": silhouette_score(X, labels)})
    table = pd.DataFrame(rows)
    best = int(table.loc[table["silhouette"].idxmax(), "k"])
    return best, table


def embed_2d(X: np.ndarray, random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray, float]:
    """PCA and UMAP 2-D coordinates plus PCA explained variance of the two components."""
    pca = PCA(n_components=2, random_state=random_state)
    pca_xy = pca.fit_transform(X)
    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=min(UMAP_N_NEIGHBORS, len(X) - 1),
            min_dist=UMAP_MIN_DIST,
            n_components=UMAP_N_COMPONENTS,
            random_state=random_state,
            n_jobs=1,
        )
        umap_xy = reducer.fit_transform(X)
    except ImportError:  # umap-learn is optional at runtime
        umap_xy = np.full((len(X), 2), np.nan)
    return pca_xy, umap_xy, float(pca.explained_variance_ratio_.sum())


def cluster_drugs(
    features: np.ndarray, drug_names, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], pd.DataFrame]:
    """
    Standardise, cluster with K-means (silhouette-chosen k), Ward and DBSCAN, and embed in 2-D.

    Returns (assignments table, metrics per method, silhouette-vs-k table).
    """
    X = StandardScaler().fit_transform(np.asarray(features, float))
    k, k_table = choose_k(X, random_state=random_state)
    kmeans = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=random_state).fit_predict(X)
    ward = AgglomerativeClustering(n_clusters=k, linkage=HIERARCHICAL_LINKAGE).fit_predict(X)
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X)
    pca_xy, umap_xy, pca_var = embed_2d(X, random_state)

    table = pd.DataFrame(
        {
            "drug_name": list(drug_names),
            "kmeans_cluster": kmeans,
            "ward_cluster": ward,
            "dbscan_cluster": dbscan,
            "pca_1": pca_xy[:, 0],
            "pca_2": pca_xy[:, 1],
            "umap_1": umap_xy[:, 0],
            "umap_2": umap_xy[:, 1],
        }
    )
    metrics = {
        "kmeans": {**cluster_metrics(X, kmeans), "k": k},
        "ward": cluster_metrics(X, ward),
        "dbscan": cluster_metrics(X, dbscan),
        "pca": {"explained_variance_2d": pca_var},
    }
    logger.info(
        "Clustering: k=%d (silhouette %.3f); DBSCAN found %d clusters, %d noise",
        k,
        metrics["kmeans"]["silhouette"],
        metrics["dbscan"]["n_clusters"],
        metrics["dbscan"]["n_noise"],
    )
    return table, metrics, k_table
