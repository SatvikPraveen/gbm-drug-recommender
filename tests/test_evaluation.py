"""Tests for the evaluation harness on synthetic data: no leakage, sane metrics, working null model."""

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
    scores, oof = ev.run_benchmark(
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
