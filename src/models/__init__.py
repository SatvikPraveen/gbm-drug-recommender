"""
Machine Learning Models Package

Implements supervised and unsupervised learning for drug analysis.

Classes:
- DrugClusteringAnalyzer: KMeans, DBSCAN, Hierarchical clustering with UMAP
- OneClassDrugPredictor: Semi-supervised learning for candidate identification
- ModelComparison: Benchmark 5 classifiers (KNN, RF, SVM, NN, XGBoost)

Usage:
    from src.models import DrugClusteringAnalyzer, OneClassDrugPredictor
    clustering = DrugClusteringAnalyzer()
    predictor = OneClassDrugPredictor()
"""

from .clustering import DrugClusteringAnalyzer
from .one_class_svm import OneClassDrugPredictor

__all__ = [
    'DrugClusteringAnalyzer',
    'OneClassDrugPredictor'
]
