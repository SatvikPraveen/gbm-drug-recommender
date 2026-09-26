"""
The models benchmarked in this project, as :class:`gbm_drug.evaluation.ModelSpec` entries.

Each spec names the feature block it consumes so the harness evaluates every
model on identical folds. Hyper-parameters are fixed, sensible defaults chosen
*a priori* (not tuned on this data); nested tuning would be the next step and is
noted as such in docs/METHODS.md. Classification models handle the ~4 % positive
rate with class weighting.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from ..config import RANDOM_STATE
from ..evaluation import ModelSpec


def _linear(kind: str) -> BaseEstimator:
    if kind == "classification":
        return make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
        )
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def _knn_jaccard(kind: str) -> BaseEstimator:
    to_bool = FunctionTransformer(lambda X: np.asarray(X, dtype=bool))
    if kind == "classification":
        return make_pipeline(
            to_bool, KNeighborsClassifier(n_neighbors=7, metric="jaccard", weights="distance")
        )
    return make_pipeline(to_bool, KNeighborsRegressor(n_neighbors=7, metric="jaccard", weights="distance"))


def _random_forest(kind: str) -> BaseEstimator:
    common = dict(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)
    if kind == "classification":
        return RandomForestClassifier(class_weight="balanced_subsample", **common)
    return RandomForestRegressor(**common)


def _xgboost(kind: str) -> BaseEstimator:
    common = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbosity=0,
    )
    if kind == "classification":
        return _PosWeightXGBClassifier(**common)
    return XGBRegressor(**common)


class _PosWeightXGBClassifier(XGBClassifier):
    """XGBClassifier that sets scale_pos_weight from the training fold's class ratio."""

    def fit(self, X, y, **kw):
        y = np.asarray(y)
        pos = max(float(y.sum()), 1.0)
        self.set_params(scale_pos_weight=(len(y) - pos) / pos)
        return super().fit(X, y, **kw)


def _svm(kind: str) -> BaseEstimator:
    if kind == "classification":
        return make_pipeline(
            StandardScaler(),
            SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        )
    return make_pipeline(StandardScaler(), SVR(C=1.0, gamma="scale", epsilon=0.1))


def _mlp(kind: str) -> BaseEstimator:
    common = dict(
        hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=2000, early_stopping=True, random_state=RANDOM_STATE
    )
    if kind == "classification":
        return make_pipeline(StandardScaler(), MLPClassifier(**common))
    return make_pipeline(StandardScaler(), MLPRegressor(**common))


def _gnn_factory(**overrides) -> Callable[[str], BaseEstimator]:
    def factory(kind: str) -> BaseEstimator:
        from .gnn_model import GNNDrugPredictor  # torch import deferred until needed

        return GNNDrugPredictor(task=kind, **overrides)

    return factory


def tabular_models() -> list[ModelSpec]:
    return [
        ModelSpec("Linear (descriptors)", _linear, "descriptors", ("linear",)),
        ModelSpec("KNN-Jaccard (morgan)", _knn_jaccard, "morgan", ("knn",)),
        ModelSpec("Random Forest (descriptors)", _random_forest, "descriptors", ("tree",)),
        ModelSpec("Random Forest (morgan)", _random_forest, "morgan", ("tree",)),
        ModelSpec("XGBoost (descriptors)", _xgboost, "descriptors", ("tree",)),
        ModelSpec("XGBoost (morgan)", _xgboost, "morgan", ("tree",)),
        ModelSpec("SVM (descriptors)", _svm, "descriptors", ("kernel",)),
        ModelSpec("MLP (descriptors)", _mlp, "descriptors", ("neural",)),
    ]


def annotation_models() -> list[ModelSpec]:
    """Models on GDSC's own pathway annotation (one-hot), to compare target biology against chemistry."""
    return [
        ModelSpec("Linear (pathway one-hot)", _linear, "pathway", ("annotation",)),
        ModelSpec("Random Forest (pathway one-hot)", _random_forest, "pathway", ("annotation",)),
        ModelSpec(
            "Random Forest (descriptors + pathway)",
            _random_forest,
            "descriptors_pathway",
            ("annotation", "tree"),
        ),
    ]


def gnn_models(repeats: int = 1, **overrides) -> list[ModelSpec]:
    return [
        ModelSpec(
            "GNN-GCN (smiles)",
            _gnn_factory(gnn_type="gcn", **overrides),
            "smiles",
            ("neural", "graph"),
            repeats=repeats,
        ),
        ModelSpec(
            "GNN-GAT (smiles)",
            _gnn_factory(gnn_type="gat", **overrides),
            "smiles",
            ("neural", "graph"),
            repeats=repeats,
        ),
    ]


def all_models(include_gnn: bool = True, gnn_repeats: int = 1, **gnn_overrides) -> list[ModelSpec]:
    specs = tabular_models() + annotation_models()
    if include_gnn:
        specs += gnn_models(repeats=gnn_repeats, **gnn_overrides)
    return specs
