"""
One-class novelty scoring, evaluated honestly.

A One-Class SVM fitted on the GBM-selective drugs learns the region of descriptor
space they occupy; other drugs are scored by how far inside that region they
fall. This is an *exploratory* ranking, kept for continuity with earlier versions
of the project, and it is weaker than the supervised models in :mod:`zoo`.

The earlier implementation scored the training drugs with the model trained on
them and reported them as "13 promising candidates". Here every drug receives an
**out-of-fold** score: positives are scored by a model that never saw them, and
the ranking quality is reported as ROC-AUC / PR-AUC of positives vs the rest.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ..config import CV_FOLDS, RANDOM_STATE, SVM_GAMMA, SVM_KERNEL, SVM_NU
from ..evaluation import classification_metrics

logger = logging.getLogger(__name__)


def _make_ocsvm(kernel: str, nu: float, gamma) -> object:
    return make_pipeline(StandardScaler(), OneClassSVM(kernel=kernel, nu=nu, gamma=gamma))


def novelty_scores(
    X: np.ndarray,
    positive: np.ndarray,
    groups: np.ndarray,
    n_splits: int = CV_FOLDS,
    kernel: str = SVM_KERNEL,
    nu: float = SVM_NU,
    gamma=SVM_GAMMA,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Out-of-fold One-Class SVM decision scores for every row.

    Positives are split into grouped folds; for each fold the model is fitted on the
    remaining positives and scores (a) the held-out positives and (b) a matching
    share of the negatives, so each row is scored exactly once by a model that did
    not train on it. Higher = more like the positive set.
    """
    X = np.asarray(X, float)
    positive = np.asarray(positive, bool)
    pos_idx = np.where(positive)[0]
    neg_idx = np.where(~positive)[0]
    if len(pos_idx) < n_splits:
        raise ValueError(
            f"need at least {n_splits} positives for {n_splits}-fold novelty scoring, got {len(pos_idx)}"
        )

    scores = np.full(len(X), np.nan)
    pos_folds = list(
        GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
            pos_idx, groups=groups[pos_idx]
        )
    )
    neg_folds = np.array_split(np.random.default_rng(random_state).permutation(neg_idx), n_splits)
    for (train_p, test_p), test_n in zip(pos_folds, neg_folds):
        model = _make_ocsvm(kernel, nu, gamma).fit(X[pos_idx[train_p]])
        held = np.concatenate([pos_idx[test_p], test_n])
        scores[held] = model.decision_function(X[held])

    # Rank-based metrics: how well do OOF scores separate positives from the rest?
    valid = ~np.isnan(scores)
    metrics = classification_metrics(positive[valid].astype(int), _to_unit(scores[valid]))
    logger.info(
        "One-class novelty: OOF ROC-AUC %.3f, PR-AUC %.3f (%d positives)",
        metrics["roc_auc"],
        metrics["pr_auc"],
        len(pos_idx),
    )
    return scores, metrics


def _to_unit(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.full_like(x, 0.5)


def novelty_table(drug_names, scores: np.ndarray, positive: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(
        {"drug_name": list(drug_names), "novelty_score": scores, "is_positive": np.asarray(positive, bool)}
    )
    out["rank"] = out["novelty_score"].rank(ascending=False, method="min").astype("Int64")
    return out.sort_values("novelty_score", ascending=False).reset_index(drop=True)
