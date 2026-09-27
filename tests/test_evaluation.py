"""Tests for the evaluation harness on synthetic data: no leakage, sane metrics, working null model."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from gbm_drug import evaluation as ev


def _synthetic(n_groups=40, per_group=3, seed=0):
    """n molecules in groups (e.g. salts of one compound); y depends linearly on x plus noise."""
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    groups = np.repeat(np.arange(n_groups), per_group)
    X = rng.normal(size=(n, 5))
    X[:, 0] += groups * 0.05  # mild group structure
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.5, size=n)
    targets = pd.DataFrame(
        {"drug_name": [f"d{i}" for i in range(n)], "y": y, "label": (y > np.median(y)).astype(int)}
    )
    return X, groups, targets


def test_connectivity_groups_merge_salts_and_stereoisomers():
    keys = ["AAAA-BBBB-N", "AAAA-CCCC-N", "DDDD-BBBB-N", None, ""]
    g = ev.connectivity_groups(keys)
    assert g[0] == g[1]
    assert g[0] != g[2]
    assert len({g[3], g[4]}) == 2  # missing keys never merge


def test_scaffold_kfold_is_balanced_and_scaffold_disjoint():
    groups = np.array([0] * 10 + [1] * 6 + [2] * 5 + [3] * 3 + [4] * 2 + [5, 6, 7, 8])
    folds = ev.scaffold_kfold(groups, 3)
    assert sorted(np.concatenate(folds)) == list(range(len(groups)))
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 10  # largest single scaffold bounds the imbalance
    for i, f in enumerate(folds):
        for j, g in enumerate(folds):
            if i != j:
                assert not set(groups[f]) & set(groups[g])


@pytest.mark.parametrize("kind", ["regression", "classification"])
def test_grouped_splits_never_leak_groups(kind):
    X, groups, targets = _synthetic()
    y = targets["label" if kind == "classification" else "y"].to_numpy()
    seen = 0
    for _, _, tr, te in ev.iter_splits(y, groups, kind, "grouped", n_splits=5, n_repeats=2):
        ev.assert_no_group_leakage(groups, tr, te)
        assert len(np.intersect1d(tr, te)) == 0
        seen += 1
    assert seen == 10


def test_assert_no_group_leakage_detects_overlap():
    groups = np.array([0, 0, 1, 1])
    with pytest.raises(AssertionError):
        ev.assert_no_group_leakage(groups, np.array([0, 2]), np.array([1, 3]))


def test_run_benchmark_beats_baseline_on_learnable_signal():
    X, groups, targets = _synthetic()
    tasks = [ev.Task("reg", "regression", "y"), ev.Task("clf", "classification", "label")]
    models = [
        ev.ModelSpec(
            "Ridge/LogReg",
            lambda k: LogisticRegression(max_iter=500) if k == "classification" else Ridge(),
            "descriptors",
        ),
    ]
    scores, oof, _ = ev.run_benchmark(
        tasks, models, {"descriptors": X}, targets, {"grouped": groups}, n_splits=4, n_repeats=2
    )
    summary = ev.summarize_scores(scores)
    # Every (task, model, metric) has 8 fold scores and a CI that brackets the mean.
    assert (summary["n_folds"] == 8).all()
    assert (summary["ci_low"] <= summary["mean"] + 1e-12).all() and (
        summary["ci_high"] >= summary["mean"] - 1e-12
    ).all()
    best = ev.best_models(summary, tasks).set_index("task")
    assert best.loc["reg", "model"] == "Ridge/LogReg"
    assert best.loc["reg", "mean"] > 0.8 > best.loc["reg", "baseline_mean"]  # spearman rho
    assert best.loc["clf", "mean"] > 0.8 > best.loc["clf", "baseline_mean"]  # roc auc
    # Out-of-fold predictions: every drug predicted exactly once per (task, model) in repeat 0.
    counts = oof.groupby(["task", "model"])["drug_name"].nunique()
    assert (counts == len(targets)).all()


def test_run_benchmark_rejects_misaligned_features():
    X, groups, targets = _synthetic()
    with pytest.raises(ValueError, match="rows"):
        ev.run_benchmark(
            [ev.Task("reg", "regression", "y")], [], {"descriptors": X[:-1]}, targets, {"grouped": groups}
        )


def test_metrics_handle_constant_predictions_and_single_class():
    r = ev.regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]))
    assert r["spearman_rho"] == 0.0 and r["pearson_r"] == 0.0
    c = ev.classification_metrics(np.array([1, 1, 1]), np.array([0.2, 0.9, 0.6]))
    assert np.isnan(c["roc_auc"]) and np.isnan(c["pr_auc"])


def test_y_scramble_gives_small_p_for_real_signal_and_large_for_noise():
    X, groups, targets = _synthetic()
    task = ev.Task("reg", "regression", "y")
    spec = ev.ModelSpec("Ridge", lambda k: Ridge(), "descriptors")
    df = ev.y_scramble(task, spec, {"descriptors": X}, targets, groups, n_rounds=10, n_splits=4)
    assert df.attrs["empirical_p"] <= 1 / 11 + 1e-9
    noise = targets.assign(y=np.random.default_rng(1).normal(size=len(targets)))
    df2 = ev.y_scramble(task, spec, {"descriptors": X}, noise, groups, n_rounds=10, n_splits=4)
    assert df2.attrs["empirical_p"] > 0.1


def test_bootstrap_ci_edge_cases():
    assert ev.bootstrap_ci(np.array([])) == (np.nan, np.nan) or all(np.isnan(ev.bootstrap_ci(np.array([]))))
    assert ev.bootstrap_ci(np.array([0.5])) == (0.5, 0.5)
    lo, hi = ev.bootstrap_ci(np.arange(10, dtype=float))
    assert lo < 4.5 < hi


def test_best_models_picks_highest_for_higher_is_better_and_lowest_for_rmse():
    summary = pd.DataFrame(
        {
            "task": ["reg"] * 6,
            "strategy": ["grouped"] * 6,
            "model": ["Baseline", "A", "B", "Baseline", "A", "B"],
            "features": ["d"] * 6,
            "metric": ["spearman_rho"] * 3 + ["rmse"] * 3,
            "mean": [0.0, 0.2, 0.5, 1.0, 0.9, 0.7],
            "ci_low": [0.0, 0.1, 0.4, 1.0, 0.8, 0.6],
            "ci_high": [0.0, 0.3, 0.6, 1.0, 1.0, 0.8],
        }
    )
    best = ev.best_models(summary, [ev.Task("reg", "regression", "y")])
    assert best.iloc[0]["model"] == "B" and best.iloc[0]["baseline_mean"] == 0.0
    # Primary metric for regression is spearman_rho; make sure a lower-is-better metric would also work.
    ev.PRIMARY_METRIC["regression"] = "rmse"
    try:
        assert ev.best_models(summary, [ev.Task("reg", "regression", "y")]).iloc[0]["model"] == "B"
    finally:
        ev.PRIMARY_METRIC["regression"] = "spearman_rho"


class _GroupSpy(Ridge):
    """Ridge that records which molecule groups each fit saw, to verify inner folds are grouped."""

    seen: list = []

    def fit(self, X, y):
        _GroupSpy.seen.append(np.asarray(X[:, -1], int).tolist())
        return super().fit(X[:, :-1], y)

    def predict(self, X):
        return super().predict(X[:, :-1])


def test_nested_tuning_is_grouped_and_records_params():
    X, groups, targets = _synthetic(n_groups=30, per_group=3)
    Xg = np.hstack([X, groups[:, None]])  # last column carries the group id for the spy
    spec = ev.ModelSpec(
        "spy", lambda k: _GroupSpy(), "descriptors", param_grid=lambda k: {"alpha": [0.1, 1.0, 10.0]}
    )
    _GroupSpy.seen = []
    scores, oof, tuning = ev.run_benchmark(
        [ev.Task("reg", "regression", "y")],
        [spec],
        {"descriptors": Xg},
        targets,
        {"grouped": groups},
        n_splits=3,
        n_repeats=1,
        include_baseline=False,
        tune=True,
    )
    # one tuning row per outer fold, with a parsable params dict and an inner score
    assert len(tuning) == 3 and set(tuning.columns) >= {
        "task",
        "strategy",
        "model",
        "fold",
        "inner_score",
        "params",
    }
    assert all(set(json.loads(p)) == {"alpha"} for p in tuning["params"])
    # Inner fits: 3 outer folds x 3 grid points x 3 inner folds = 27, then 3 refits on the full training fold.
    assert len(_GroupSpy.seen) == 27 + 3
    # Every inner fit's training groups are disjoint from the groups it is later scored on: check that
    # within each outer fold no inner-train set equals the outer-train set except the refit (last of each block).
    outer_train_sets = [set(s) for s in _GroupSpy.seen[9::10]]
    inner_sets = [set(s) for i, s in enumerate(_GroupSpy.seen) if i % 10 != 9]
    assert all(inner < outer_train_sets[i // 9] for i, inner in enumerate(inner_sets))


def test_build_model_without_grid_or_without_tune_is_plain_estimator():
    spec = ev.ModelSpec("r", lambda k: Ridge(), "descriptors")
    assert isinstance(ev.build_model(spec, "regression", tune=True), Ridge)
    spec2 = ev.ModelSpec("r", lambda k: Ridge(), "descriptors", param_grid=lambda k: {"alpha": [1.0, 2.0]})
    assert isinstance(ev.build_model(spec2, "regression", tune=False), Ridge)
    assert isinstance(ev.build_model(spec2, "regression", tune=True), ev.TunedModel)


def test_tuned_model_classification_uses_stratified_grouped_inner_cv_and_exposes_proba():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    X = rng.normal(size=(90, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=90) > 0).astype(int)
    groups = np.repeat(np.arange(30), 3)
    m = ev.TunedModel(LogisticRegression(max_iter=500), {"C": [0.1, 1.0]}, "classification").fit(
        X, y, groups=groups
    )
    assert m.best_params_["C"] in (0.1, 1.0)
    assert m.predict_proba(X).shape == (90, 2)
    assert 0 <= m.best_inner_score_ <= 1
