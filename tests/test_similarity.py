import numpy as np
import pandas as pd
import pytest

from gbm_drug.similarity import (
    mantel_test,
    mcs_matrix,
    mcs_similarity,
    pairs_above,
    summarize_matrix,
    tanimoto_matrix,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SALICYLIC = "OC(=O)c1ccccc1O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
BENZENE = "c1ccccc1"
ETHANE = "CC"

SMILES = {"aspirin": ASPIRIN, "salicylic": SALICYLIC, "caffeine": CAFFEINE, "benzene": BENZENE}


def test_tanimoto_matrix_properties():
    m = tanimoto_matrix(SMILES)
    assert list(m.index) == list(SMILES)
    np.testing.assert_allclose(np.diag(m), 1.0)
    np.testing.assert_allclose(m.to_numpy(), m.to_numpy().T)
    assert m.loc["aspirin", "salicylic"] > m.loc["aspirin", "caffeine"]
    assert ((m >= 0) & (m <= 1)).all().all()


def test_tanimoto_invalid_smiles_gives_nan_not_zero():
    m = tanimoto_matrix({"ok": ASPIRIN, "bad": "C1CC"})
    assert np.isnan(m.loc["ok", "bad"]) and np.isnan(m.loc["bad", "bad"])
    assert m.loc["ok", "ok"] == 1.0


def test_pairs_above():
    m = tanimoto_matrix(SMILES)
    p = pairs_above(m, 0.2)
    assert {"drug_a", "drug_b", "similarity"} <= set(p.columns)
    assert p.iloc[0]["similarity"] == p["similarity"].max()
    assert set(map(frozenset, zip(p["drug_a"], p["drug_b"]))) == {frozenset({"aspirin", "salicylic"})}


def test_mcs_similarity_known_values():
    assert mcs_similarity(ASPIRIN, ASPIRIN) == 1.0
    # salicylic acid (10 heavy atoms) is a substructure of aspirin (13): 10 / (13 + 10 - 10)
    assert mcs_similarity(ASPIRIN, SALICYLIC) == pytest.approx(10 / 13)
    # benzene is entirely contained in aspirin: 6 / (13 + 6 - 6)
    assert mcs_similarity(ASPIRIN, BENZENE) == pytest.approx(6 / 13)
    # ring atoms only match ring atoms and rings must be complete, so ethane shares nothing with benzene
    assert mcs_similarity(ETHANE, BENZENE) == 0.0
    assert np.isnan(mcs_similarity(ASPIRIN, "C1CC"))


def test_mcs_matrix_is_not_all_zero_and_symmetric():
    m = mcs_matrix(SMILES, n_jobs=1)
    np.testing.assert_allclose(np.diag(m), 1.0)
    np.testing.assert_allclose(m.to_numpy(), m.to_numpy().T)
    off = m.to_numpy()[np.triu_indices(len(m), k=1)]
    assert (off > 0).sum() >= 3  # the old backend produced an identity matrix


def test_mcs_matrix_raises_on_mostly_invalid_input():
    with pytest.raises(RuntimeError, match="failed"):
        mcs_matrix({"a": "C1CC", "b": "C1CC", "c": ASPIRIN}, n_jobs=1)


def test_mantel_test_detects_shared_structure_and_not_noise():
    rng = np.random.default_rng(0)
    names = [f"d{i}" for i in range(15)]
    base = rng.uniform(size=(15, 15))
    base = (base + base.T) / 2
    np.fill_diagonal(base, 1)
    a = pd.DataFrame(base, index=names, columns=names)
    b = pd.DataFrame(
        np.clip(base + rng.normal(scale=0.05, size=base.shape), 0, 1), index=names, columns=names
    )
    b = (b + b.T) / 2
    res = mantel_test(a, b, n_permutations=199)
    assert res["r"] > 0.8 and res["p"] < 0.01 and res["n_drugs"] == 15
    noise = rng.uniform(size=(15, 15))
    c = pd.DataFrame((noise + noise.T) / 2, index=names, columns=names)
    assert mantel_test(a, c, n_permutations=199)["p"] > 0.05


def test_mantel_aligns_on_shared_drugs_and_needs_enough():
    names = [f"d{i}" for i in range(6)]
    a = pd.DataFrame(np.eye(6), index=names, columns=names)
    b = pd.DataFrame(np.eye(5), index=names[:5], columns=names[:5])
    assert mantel_test(a, b, n_permutations=9)["n_drugs"] == 5
    with pytest.raises(ValueError):
        mantel_test(a.iloc[:3, :3], b, n_permutations=9)


def test_summarize_matrix():
    m = tanimoto_matrix(SMILES)
    s = summarize_matrix(m, 0.5)
    assert s["n_drugs"] == 4 and s["n_pairs"] == 6 and 0 <= s["mean"] <= 1
