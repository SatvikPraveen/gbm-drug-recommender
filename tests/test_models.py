import numpy as np
import pandas as pd
import pytest

from gbm_drug import evaluation as ev
from gbm_drug.models import all_models, cluster_drugs, novelty_scores, novelty_table, tabular_models
from gbm_drug.models.zoo import annotation_models


def _tabular_data(n=90, seed=0):
    rng = np.random.default_rng(seed)
    desc = rng.normal(size=(n, 6))
    morgan = (rng.uniform(size=(n, 64)) < 0.15).astype(np.uint8)
    y = 1.5 * desc[:, 0] + morgan[:, :8].sum(axis=1) * 0.3 + rng.normal(scale=0.3, size=n)
    label = (y > np.quantile(y, 0.8)).astype(int)  # ~20 % positives
    targets = pd.DataFrame({"drug_name": [f"d{i}" for i in range(n)], "y": y, "label": label})
    groups = np.arange(n)
    return {"descriptors": desc, "morgan": morgan}, targets, groups


@pytest.mark.parametrize("kind", ["regression", "classification"])
def test_every_tabular_model_fits_and_predicts(kind):
    feats, targets, _ = _tabular_data()
    y = targets["label" if kind == "classification" else "y"].to_numpy()
    for spec in tabular_models():
        model = spec.factory(kind)
        X = feats[spec.features]
        model.fit(X[:60], y[:60])
        if kind == "classification":
            # The harness scores classifiers via predict_proba or, failing that, decision_function.
            score = ev._predict(model, kind, X[60:])
            assert score.shape == (30,) and np.isfinite(score).all()
        else:
            assert model.predict(X[60:]).shape == (30,)


def test_all_models_includes_gnn_specs_with_repeat_override():
    specs = all_models(include_gnn=True, gnn_repeats=1)
    gnn = [s for s in specs if "GNN" in s.name]
    assert len(gnn) == 2 and all(s.features == "smiles" and s.repeats == 1 for s in gnn)
    assert len(all_models(include_gnn=False)) == len(tabular_models()) + len(annotation_models())


def test_tabular_models_beat_baseline_in_harness():
    feats, targets, groups = _tabular_data()
    tasks = [ev.Task("reg", "regression", "y")]
    specs = [s for s in tabular_models() if s.name.startswith(("Linear", "Random Forest (desc"))]
    scores, _, _ = ev.run_benchmark(
        tasks, specs, feats, targets, {"grouped": groups}, n_splits=3, n_repeats=1
    )
    summary = ev.summarize_scores(scores)
    rho = summary[summary["metric"] == "spearman_rho"].set_index("model")["mean"]
    assert rho["Linear (descriptors)"] > 0.6 > rho["Baseline"]


def test_novelty_scores_are_out_of_fold_and_rank_positives_higher():
    rng = np.random.default_rng(1)
    n = 120
    X = rng.normal(size=(n, 4))
    positive = np.zeros(n, bool)
    positive[:30] = True
    X[positive] += 3.0  # positives live in a shifted region
    scores, metrics = novelty_scores(X, positive, groups=np.arange(n), n_splits=5)
    assert np.isfinite(scores).all()
    assert metrics["roc_auc"] > 0.9
    table = novelty_table([f"d{i}" for i in range(n)], scores, positive)
    assert table.iloc[0]["rank"] == 1 and table["is_positive"].head(10).mean() > 0.8


def test_novelty_scores_needs_enough_positives():
    with pytest.raises(ValueError):
        novelty_scores(np.zeros((10, 2)), np.array([True] * 2 + [False] * 8), np.arange(10), n_splits=5)


def test_cluster_drugs_returns_assignments_and_metrics():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(loc=c, scale=0.3, size=(25, 5)) for c in (-4, 0, 4)])
    names = [f"d{i}" for i in range(len(X))]
    table, metrics, k_table = cluster_drugs(X, names)
    assert len(table) == 75 and {
        "kmeans_cluster",
        "ward_cluster",
        "dbscan_cluster",
        "pca_1",
        "umap_1",
    } <= set(table.columns)
    assert metrics["kmeans"]["k"] == 3 and metrics["kmeans"]["silhouette"] > 0.7
    assert set(k_table.columns) == {"k", "silhouette"}


def test_every_model_spec_is_picklable_after_fit(tmp_path):
    """final_models persists fitted estimators with joblib; lambdas inside pipelines would break that."""
    import io

    import joblib

    feats, targets, _ = _tabular_data(n=40)
    y = targets["y"].to_numpy()
    feats["pathway"] = (np.arange(40) % 3)[:, None] == np.arange(3)[None, :]
    feats["descriptors_pathway"] = np.hstack([feats["descriptors"], feats["pathway"].astype(np.uint8)])
    for spec in all_models(include_gnn=False):
        model = spec.factory("regression").fit(feats[spec.features], y)
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        assert joblib.load(buf).predict(feats[spec.features][:3]).shape == (3,)
