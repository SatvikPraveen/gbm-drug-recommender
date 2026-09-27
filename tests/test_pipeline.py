"""Pipeline wiring tests. The feature-stage test uses the committed data tables (no network)."""

import numpy as np
import pytest

from gbm_drug import config as cfg
from gbm_drug import pipeline as pl


def test_resolve_stages_adds_dependencies_in_order():
    assert pl.resolve_stages(["combinations"]) == ["data", "features", "combinations"]
    assert pl.resolve_stages(["report"]) == ["data", "report"]
    # final_models loads benchmark artifacts from disk, so benchmark is not a hard dependency
    assert pl.resolve_stages(["final_models", "novelty"]) == ["data", "features", "final_models", "novelty"]
    assert pl.resolve_stages(list(pl.STAGES)) == list(pl.STAGES)


def test_resolve_stages_rejects_unknown():
    with pytest.raises(ValueError, match="unknown stage"):
        pl.resolve_stages(["nope"])


def test_quick_option_disables_expensive_settings():
    o = pl.Options(quick=True)
    assert not o.include_gnn and o.cv_repeats == 1 and o.mcs_max_drugs <= 40


@pytest.mark.skipif(
    not (cfg.DRUG_SUMMARY_FILE.exists() and cfg.SMILES_FILE.exists()), reason="committed data tables missing"
)
def test_data_and_feature_stages_on_committed_tables():
    ctx = pl.Context(options=pl.Options(quick=True))
    pl.stage_data(ctx)
    assert ctx.dose_response_counts["n_drugs"] >= 500
    assert ctx.dose_response_counts["n_cell_lines"] >= 30
    pl.stage_features(ctx)
    t = ctx.targets
    assert len(t) >= 400
    assert t["drug_name"].is_unique
    assert ctx.features["descriptors"].shape == (len(t), len(cfg.MOLECULAR_DESCRIPTORS))
    assert ctx.features["morgan"].shape == (len(t), cfg.FINGERPRINT_BITS)
    assert len(ctx.features["smiles"]) == len(t)
    assert not np.isnan(ctx.features["descriptors"]).any()
    # Bleomycin screened at two concentrations shares one connectivity group.
    names = t["drug_name"].tolist()
    if "Bleomycin (10 uM)" in names and "Bleomycin (50 uM)" in names:
        g = ctx.groups["grouped"]
        assert g[names.index("Bleomycin (10 uM)")] == g[names.index("Bleomycin (50 uM)")]
    # Grouped and scaffold groupings are both valid partitions of the drug set.
    for key in ("grouped", "scaffold"):
        assert len(ctx.groups[key]) == len(t)
    assert len(set(ctx.groups["scaffold"])) < len(t)  # scaffolds are shared


def test_candidate_set_prefers_selective_then_top_n():
    ctx = pl.Context(options=pl.Options(quick=True, top_n=3, mcs_max_drugs=4))
    import pandas as pd

    ctx.targets = pd.DataFrame(
        {
            "drug_name": list("abcdef"),
            "z_mean": [0.5, -0.9, -0.1, -0.8, -0.7, 0.1],
            "gbm_selective": [False, True, False, False, False, True],
        }
    )
    cand = pl._candidate_set(ctx)
    assert cand[:2] == ["b", "f"]  # selective first (in table order)
    assert len(cand) == 4  # capped
    assert "d" in cand  # among top-3 by z_mean


def test_benchmark_and_final_model_stages_run_end_to_end(tmp_path, monkeypatch):
    """Exercise stage_benchmark -> stage_final_models -> stage_novelty on a tiny synthetic context with tuning on."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression, Ridge

    from gbm_drug import evaluation as ev

    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 4))
    z = X[:, 0] + rng.normal(scale=0.3, size=n)
    targets = pd.DataFrame(
        {
            "drug_name": [f"d{i}" for i in range(n)],
            "putative_target": "EGFR",
            "pathway_name": "EGFR signaling",
            "n_cell_lines": 30,
            "z_mean": z,
            "z_q": 0.5,
            "ln_ic50_mean": -z,
            "ic50_um_geomean": 1.0,
            "gbm_selective": z < np.quantile(z, 0.2),
            "potent": False,
            "scaffold": "c1ccccc1",
            "inchikey_parent": [f"K{i}-X-N" for i in range(n)],
            "smiles_parent": "c1ccccc1",
        }
    )
    ctx = pl.Context(options=pl.Options(quick=True, scramble_rounds=2, cv_repeats=1))
    ctx.options.tune = True
    ctx.targets = targets
    ctx.features = {
        "descriptors": X,
        "morgan": (X > 0).astype(np.uint8),
        "smiles": ["c1ccccc1"] * n,
        "pathway": np.ones((n, 1)),
        "descriptors_pathway": X,
    }
    ctx.groups = {"grouped": np.arange(n), "scaffold": np.arange(n) % 5}
    spec = ev.ModelSpec(
        "Linear (descriptors)",
        lambda k: LogisticRegression(max_iter=500) if k == "classification" else Ridge(),
        "descriptors",
        param_grid=lambda k: {"C": [0.1, 1.0]} if k == "classification" else {"alpha": [0.1, 10.0]},
    )
    monkeypatch.setattr(pl, "_models", lambda c: [spec])
    monkeypatch.setattr(pl, "tabular_models", lambda: [spec])
    monkeypatch.setattr(pl.cfg, "BENCHMARK_RESULTS_DIR", tmp_path / "benchmark")
    monkeypatch.setattr(pl.cfg, "MODEL_RESULTS_DIR", tmp_path / "models")
    monkeypatch.setattr(pl.cfg, "RESULTS_DIR", tmp_path)

    pl.stage_benchmark(ctx)
    assert (tmp_path / "benchmark" / "summary.csv").exists()
    assert (tmp_path / "benchmark" / "tuning.csv").exists() and (
        tmp_path / "benchmark" / "tuning_summary.csv"
    ).exists()
    assert set(ctx.scramble_p) == {t.name for t in pl.TASKS}
    pl.stage_final_models(ctx)
    assert (tmp_path / "models" / "drug_scores.csv").exists()
    assert all(isinstance(m, ev.TunedModel) for _, m in ctx.final_models.values())
    pl.stage_novelty(ctx)
    assert (tmp_path / "models" / "novelty_scores.csv").exists()
