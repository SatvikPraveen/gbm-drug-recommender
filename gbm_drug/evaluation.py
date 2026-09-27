"""
Leakage-free benchmarking of structure -> GBM-response models.

Design decisions (see docs/METHODS.md for the reasoning):

* **One row per molecule.** Targets are drug-level aggregates over GBM cell lines,
  so there is exactly one feature vector per molecule and no cell-line rows to
  leak between folds.
* **Grouped folds.** Folds are built over *molecule groups* (InChIKey connectivity
  block, so salts / stereoisomers / re-screens of the same compound stay
  together) or over *Bemis–Murcko scaffolds* (harder, out-of-distribution).
* **Repeats and uncertainty.** Grouped CV is repeated with different seeds; we
  report mean, SD and a percentile-bootstrap CI over fold scores.
* **Baselines and null models.** Every task includes a dummy predictor, and the
  best model is re-evaluated under y-scrambling to give an empirical p-value that
  its score exceeds chance.

The harness is representation-agnostic: each model declares which feature block
it consumes (``"descriptors"``, ``"morgan"``, ``"smiles"``), so the GNN, which
takes SMILES, is evaluated on exactly the same folds as the tabular models.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold

from .config import (
    BOOTSTRAP_SAMPLES,
    CV_FOLDS,
    CV_REPEATS,
    RANDOM_STATE,
    TUNING_INNER_FOLDS,
    Y_SCRAMBLE_ROUNDS,
)

logger = logging.getLogger(__name__)

REGRESSION_METRICS = ("rmse", "mae", "r2", "pearson_r", "spearman_rho")
CLASSIFICATION_METRICS = ("roc_auc", "pr_auc", "balanced_accuracy", "mcc", "brier")
PRIMARY_METRIC = {"regression": "spearman_rho", "classification": "roc_auc"}
HIGHER_IS_BETTER = {"rmse": False, "mae": False, "brier": False}


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    name: str
    kind: str  # "regression" | "classification"
    target: str  # column in the drug summary
    description: str = ""

    def __post_init__(self):
        if self.kind not in ("regression", "classification"):
            raise ValueError(f"kind must be 'regression' or 'classification', got {self.kind!r}")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[str], BaseEstimator]  # kind -> unfitted estimator
    features: str  # key into the feature dict
    tags: tuple[str, ...] = field(default_factory=tuple)
    repeats: int | None = None  # override CV_REPEATS (e.g. 1 for slow GNNs)
    param_grid: Callable[[str], dict] | None = None  # kind -> search space for nested tuning (None = fixed)


# ---------------------------------------------------------------------------
# Grouping and splitting
# ---------------------------------------------------------------------------


def connectivity_groups(inchikeys: Sequence[str]) -> np.ndarray:
    """Group id per molecule from the first (connectivity) block of the InChIKey."""
    blocks = [
        str(k).split("-")[0] if isinstance(k, str) and k else f"__missing_{i}"
        for i, k in enumerate(inchikeys)
    ]
    return pd.factorize(pd.Series(blocks))[0]


def scaffold_groups(scaffolds: Sequence[str]) -> np.ndarray:
    """Group id per molecule from its Bemis–Murcko scaffold; acyclic molecules each form their own group."""
    labels = [s if isinstance(s, str) and s else f"__acyclic_{i}" for i, s in enumerate(scaffolds)]
    return pd.factorize(pd.Series(labels))[0]


def scaffold_kfold(groups: np.ndarray, n_splits: int) -> list[np.ndarray]:
    """
    Deterministic k-fold assignment of scaffold groups, largest scaffolds first, each
    to the currently smallest fold (DeepChem-style balanced scaffold split generalised to k folds).
    Returns a list of test-index arrays.
    """
    sizes = pd.Series(groups).value_counts()  # sorted by size desc, then by label for ties
    order = sorted(sizes.index, key=lambda g: (-sizes[g], g))
    fold_of_group: dict[int, int] = {}
    fold_sizes = np.zeros(n_splits, dtype=int)
    for g in order:
        f = int(np.argmin(fold_sizes))
        fold_of_group[g] = f
        fold_sizes[f] += sizes[g]
    fold_ids = np.array([fold_of_group[g] for g in groups])
    return [np.where(fold_ids == f)[0] for f in range(n_splits)]


def iter_splits(
    y: np.ndarray,
    groups: np.ndarray,
    kind: str,
    strategy: str,
    n_splits: int = CV_FOLDS,
    n_repeats: int = CV_REPEATS,
    random_state: int = RANDOM_STATE,
):
    """
    Yield (repeat, fold, train_idx, test_idx).

    strategy="grouped": shuffled (Stratified)GroupKFold, one shuffle per repeat.
    strategy="scaffold": deterministic balanced scaffold folds; repeats are ignored (yields once).
    """
    n = len(y)
    if strategy == "scaffold":
        for fold, test_idx in enumerate(scaffold_kfold(groups, n_splits)):
            train_idx = np.setdiff1d(np.arange(n), test_idx)
            yield 0, fold, train_idx, test_idx
        return
    if strategy != "grouped":
        raise ValueError(f"unknown split strategy {strategy!r}")
    for repeat in range(n_repeats):
        seed = random_state + repeat
        if kind == "classification":
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(n), y, groups)):
            yield repeat, fold, train_idx, test_idx


def assert_no_group_leakage(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    if overlap:
        raise AssertionError(f"{len(overlap)} molecule group(s) appear in both train and test folds")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    out = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if np.ptp(y_pred) > 1e-12 and np.ptp(y_true) > 1e-12:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
            out["pearson_r"] = float(np.nan_to_num(stats.pearsonr(y_true, y_pred)[0]))
            out["spearman_rho"] = float(np.nan_to_num(stats.spearmanr(y_true, y_pred)[0]))
    else:  # constant predictions (e.g. the mean baseline) have undefined correlation; report 0
        out["pearson_r"] = 0.0
        out["spearman_rho"] = 0.0
    return out


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_true = np.asarray(y_true, int)
    y_score = np.asarray(y_score, float)
    y_hat = (y_score >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        roc, pr = np.nan, np.nan
    else:
        roc = float(roc_auc_score(y_true, y_score))
        pr = float(average_precision_score(y_true, y_score))
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 0, 1))),
    }


def _predict(model, kind: str, X):
    if kind == "classification":
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X))[:, 1]
        if hasattr(model, "decision_function"):
            d = np.asarray(model.decision_function(X), float)
            return 1 / (1 + np.exp(-d))
        return np.asarray(model.predict(X), float)
    return np.asarray(model.predict(X), float).ravel()


def _subset(block, idx: np.ndarray):
    if isinstance(block, np.ndarray):
        return block[idx]
    if isinstance(block, pd.DataFrame):
        return block.iloc[idx]
    return [block[i] for i in idx]  # list of SMILES


# ---------------------------------------------------------------------------
# Nested hyper-parameter tuning
# ---------------------------------------------------------------------------


def _spearman_score(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if np.ptp(y_pred) < 1e-12 or np.ptp(y_true) < 1e-12:
        return 0.0
    return float(np.nan_to_num(stats.spearmanr(y_true, y_pred)[0]))


TUNING_SCORING = {"regression": make_scorer(_spearman_score), "classification": "roc_auc"}


class TunedModel:
    """
    Nested cross-validation wrapper: picks hyper-parameters with an inner *grouped* CV on
    the training fold only, then refits the best setting on the whole training fold.

    The inner splitter groups by the same molecule ids as the outer one, so a molecule's
    salts or re-screens never sit on both sides of an inner split either. ``best_params_``
    and ``best_inner_score_`` are recorded per outer fold so the chosen settings are auditable.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: dict,
        kind: str,
        inner_folds: int = TUNING_INNER_FOLDS,
        random_state: int = RANDOM_STATE,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.kind = kind
        self.inner_folds = inner_folds
        self.random_state = random_state

    def fit(self, X, y, groups: np.ndarray | None = None):
        n_groups = len(np.unique(groups)) if groups is not None else len(y)
        k = max(2, min(self.inner_folds, n_groups))
        if self.kind == "classification":
            inner = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        else:
            inner = GroupKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        search = GridSearchCV(
            self.estimator,
            self.param_grid,
            scoring=TUNING_SCORING[self.kind],
            cv=inner,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(X, y, groups=groups if groups is not None else np.arange(len(y)))
        self.search_ = search
        self.model_ = search.best_estimator_
        self.best_params_ = dict(search.best_params_)
        self.best_inner_score_ = float(search.best_score_)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    # predict_proba / decision_function are delegated dynamically rather than defined here, so
    # hasattr() on the wrapper answers the same as on the refit estimator (an SVC without
    # probability has no predict_proba, and the harness must fall back to decision_function).
    _DELEGATED = ("predict_proba", "decision_function", "predict_log_proba")

    def __getattr__(self, name):
        inner = self.__dict__.get("model_")
        if inner is not None and (name in self._DELEGATED or name.endswith("_")):
            return getattr(inner, name)
        raise AttributeError(name)


def build_model(spec: ModelSpec, kind: str, tune: bool, random_state: int = RANDOM_STATE) -> BaseEstimator:
    """Unfitted estimator for a spec; wrapped for nested tuning when requested and a grid exists."""
    est = spec.factory(kind)
    if tune and spec.param_grid is not None:
        grid = spec.param_grid(kind)
        if grid:
            return TunedModel(est, grid, kind, random_state=random_state)
    return est


def fit_model(model, X, y, groups: np.ndarray | None = None):
    """Fit, passing molecule groups to tuned models so their inner CV is grouped too."""
    if isinstance(model, TunedModel):
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def default_baseline(kind: str) -> BaseEstimator:
    return DummyClassifier(strategy="prior") if kind == "classification" else DummyRegressor(strategy="mean")


def run_benchmark(
    tasks: Iterable[Task],
    models: Iterable[ModelSpec],
    features: Mapping[str, object],
    targets: pd.DataFrame,
    groups_by_strategy: Mapping[str, np.ndarray],
    n_splits: int = CV_FOLDS,
    n_repeats: int = CV_REPEATS,
    random_state: int = RANDOM_STATE,
    include_baseline: bool = True,
    tune: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cross-validate every model on every task under every split strategy.

    With ``tune=True`` every model that declares a ``param_grid`` is tuned by nested,
    molecule-grouped CV inside each outer training fold (see :class:`TunedModel`).

    Returns (fold_scores, oof_predictions, tuning). ``fold_scores`` is long-form with one
    row per (task, strategy, model, repeat, fold, metric); ``oof_predictions`` holds
    out-of-fold predictions from repeat 0 for every drug, model and task; ``tuning`` has
    one row per tuned outer fold with the chosen hyper-parameters and inner score.
    """
    tasks = list(tasks)
    models = list(models)
    if include_baseline:
        models = [ModelSpec("Baseline", default_baseline, "descriptors", ("baseline",)), *models]
    n = len(targets)
    for key, block in features.items():
        if len(block) != n:
            raise ValueError(f"feature block {key!r} has {len(block)} rows but targets has {n}")

    rows, preds, tuned = [], [], []
    for task in tasks:
        y_all = targets[task.target].to_numpy()
        valid = ~pd.isna(y_all)
        if task.kind == "classification":
            y_all = y_all.astype(int)
        else:
            y_all = y_all.astype(float)
        for strategy, groups in groups_by_strategy.items():
            for spec in models:
                block = features[spec.features]
                repeats = spec.repeats if spec.repeats is not None else n_repeats
                for repeat, fold, tr, te in iter_splits(
                    y_all, groups, task.kind, strategy, n_splits, repeats, random_state
                ):
                    tr = tr[valid[tr]]
                    te = te[valid[te]]
                    assert_no_group_leakage(groups, tr, te)
                    model = build_model(spec, task.kind, tune, random_state)
                    fit_model(model, _subset(block, tr), y_all[tr], groups[tr])
                    if isinstance(model, TunedModel):
                        tuned.append(
                            {
                                "task": task.name,
                                "strategy": strategy,
                                "model": spec.name,
                                "repeat": repeat,
                                "fold": fold,
                                "inner_score": model.best_inner_score_,
                                "params": json.dumps(model.best_params_, default=str),
                            }
                        )
                    score = _predict(model, task.kind, _subset(block, te))
                    metrics = (
                        classification_metrics(y_all[te], score)
                        if task.kind == "classification"
                        else regression_metrics(y_all[te], score)
                    )
                    for metric, value in metrics.items():
                        rows.append(
                            {
                                "task": task.name,
                                "strategy": strategy,
                                "model": spec.name,
                                "features": spec.features,
                                "repeat": repeat,
                                "fold": fold,
                                "metric": metric,
                                "value": value,
                            }
                        )
                    if repeat == 0:
                        for i, s in zip(te, score):
                            preds.append(
                                {
                                    "task": task.name,
                                    "strategy": strategy,
                                    "model": spec.name,
                                    "drug_name": targets.iloc[i]["drug_name"],
                                    "y_true": y_all[i],
                                    "y_pred": s,
                                    "fold": fold,
                                }
                            )
                logger.info("%-28s %-10s %-22s done", task.name, strategy, spec.name)
    return pd.DataFrame(rows), pd.DataFrame(preds), pd.DataFrame(tuned)


def bootstrap_ci(
    values: np.ndarray, n_boot: int = BOOTSTRAP_SAMPLES, alpha: float = 0.05, seed: int = RANDOM_STATE
) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize_scores(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD, n and 95 % bootstrap CI over fold scores, per (task, strategy, model, metric)."""
    keys = ["task", "strategy", "model", "features", "metric"]

    def _agg(g: pd.DataFrame) -> pd.Series:
        v = g["value"].to_numpy(float)
        lo, hi = bootstrap_ci(v)
        return pd.Series(
            {
                "mean": np.nanmean(v),
                "sd": np.nanstd(v, ddof=1) if len(v) > 1 else 0.0,
                "n_folds": len(v),
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    return fold_scores.groupby(keys).apply(_agg, include_groups=False).reset_index()


def best_models(summary: pd.DataFrame, tasks: Iterable[Task]) -> pd.DataFrame:
    """Best non-baseline model per (task, strategy) by the task's primary metric."""
    kinds = {t.name: t.kind for t in tasks}
    out = []
    for (task, strategy), g in summary.groupby(["task", "strategy"]):
        metric = PRIMARY_METRIC[kinds[task]]
        cand = g[(g["metric"] == metric) & (g["model"] != "Baseline")]
        if cand.empty:
            continue
        higher = HIGHER_IS_BETTER.get(metric, True)
        best = cand.loc[cand["mean"].idxmax()] if higher else cand.loc[cand["mean"].idxmin()]
        base = g[(g["metric"] == metric) & (g["model"] == "Baseline")]
        out.append(
            {
                "task": task,
                "strategy": strategy,
                "metric": metric,
                "model": best["model"],
                "features": best["features"],
                "mean": best["mean"],
                "ci_low": best["ci_low"],
                "ci_high": best["ci_high"],
                "baseline_mean": float(base["mean"].iloc[0]) if len(base) else np.nan,
            }
        )
    return pd.DataFrame(out)


def y_scramble(
    task: Task,
    spec: ModelSpec,
    features: Mapping[str, object],
    targets: pd.DataFrame,
    groups: np.ndarray,
    strategy: str = "grouped",
    n_rounds: int = Y_SCRAMBLE_ROUNDS,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    tune: bool = False,
) -> pd.DataFrame:
    """
    Null distribution of the primary metric when targets are permuted across molecules.
    The same procedure (including nested tuning when ``tune``) is applied to real and permuted labels.

    Returns one row per round with the mean CV score under permutation, plus the real score
    (round = -1). The empirical p-value is the fraction of null rounds >= the real score.
    """
    metric = PRIMARY_METRIC[task.kind]
    rng = np.random.default_rng(random_state)
    block = features[spec.features]
    y_real = targets[task.target].to_numpy()
    valid = ~pd.isna(y_real)
    y_real = y_real.astype(int if task.kind == "classification" else float)

    def _cv_score(y: np.ndarray) -> float:
        vals = []
        for _, _, tr, te in iter_splits(y, groups, task.kind, strategy, n_splits, 1, random_state):
            tr, te = tr[valid[tr]], te[valid[te]]
            model = build_model(spec, task.kind, tune, random_state)
            fit_model(model, _subset(block, tr), y[tr], groups[tr])
            score = _predict(model, task.kind, _subset(block, te))
            m = (
                classification_metrics(y[te], score)
                if task.kind == "classification"
                else regression_metrics(y[te], score)
            )
            vals.append(m[metric])
        return float(np.nanmean(vals))

    rows = [
        {
            "task": task.name,
            "model": spec.name,
            "strategy": strategy,
            "round": -1,
            "metric": metric,
            "value": _cv_score(y_real),
        }
    ]
    for r in range(n_rounds):
        y_perm = y_real.copy()
        y_perm[valid] = rng.permutation(y_real[valid])
        rows.append(
            {
                "task": task.name,
                "model": spec.name,
                "strategy": strategy,
                "round": r,
                "metric": metric,
                "value": _cv_score(y_perm),
            }
        )
    df = pd.DataFrame(rows)
    real = df.loc[df["round"] == -1, "value"].iloc[0]
    null = df.loc[df["round"] >= 0, "value"].to_numpy()
    df.attrs["empirical_p"] = float((np.sum(null >= real) + 1) / (len(null) + 1))
    logger.info(
        "y-scrambling %s / %s: real %s=%.3f, null mean %.3f, p=%.3f",
        task.name,
        spec.name,
        metric,
        real,
        null.mean(),
        df.attrs["empirical_p"],
    )
    return df


def fit_final_model(
    spec: ModelSpec,
    task: Task,
    features: Mapping[str, object],
    targets: pd.DataFrame,
    groups: np.ndarray | None = None,
    tune: bool = False,
) -> BaseEstimator:
    """Fit one model on every labelled molecule (for downstream scoring / embeddings).

    With ``tune`` the hyper-parameters are chosen by grouped inner CV over all labelled
    molecules, the same procedure the benchmark used inside each outer fold.
    """
    y = targets[task.target].to_numpy()
    valid = np.where(~pd.isna(y))[0]
    y = y[valid].astype(int if task.kind == "classification" else float)
    model = build_model(spec, task.kind, tune)
    fit_model(model, _subset(features[spec.features], valid), y, None if groups is None else groups[valid])
    return model


def clone_estimator(estimator: BaseEstimator) -> BaseEstimator:
    return clone(estimator)
