import numpy as np
import pytest
from rdkit import Chem

from gbm_drug.models import gnn_model as gm

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "CCO",
    "c1ccccc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1CCC[C@H]1c1cccnc1",
    "C1CCCCC1",
    "OC(=O)c1ccccc1O",
    "CCN(CC)CCOC(=O)c1ccc(N)cc1",
    "Clc1ccc(cc1)C(c1ccccc1)N1CCN(CC1)CCOCC(=O)O",
    "N.N.Cl[Pt]Cl",
    "CC(=O)Nc1ccc(O)cc1",
]


def test_smiles_to_graph_shapes_and_symmetry():
    g = gm.smiles_to_graph("CC(=O)O")
    assert g.x.shape == (4, gm.NUM_ATOM_FEATURES)
    assert g.edge_index.shape == (2, 6)  # 3 bonds, both directions
    assert g.edge_attr.shape == (6, gm.NUM_BOND_FEATURES)
    # every edge has its reverse
    edges = set(map(tuple, g.edge_index.t().tolist()))
    assert all((j, i) in edges for i, j in edges)


def test_smiles_to_graph_handles_single_atom_and_invalid():
    g = gm.smiles_to_graph("[Na+]")
    assert g.x.shape[0] == 1 and g.edge_index.shape == (2, 0)
    assert gm.smiles_to_graph("C1CC") is None


def test_atom_features_other_slot_for_unlisted_element():
    atom = Chem.MolFromSmiles("[Xe]").GetAtomWithIdx(0)
    feats = gm.atom_features(atom)
    assert feats[len(gm.ATOM_TYPES)] == 1  # trailing 'other' slot of the element one-hot


@pytest.mark.slow
@pytest.mark.parametrize("gnn_type", ["gcn", "gat"])
def test_regression_fit_predict_embed(gnn_type):
    y = np.arange(len(SMILES), dtype=float)
    est = gm.GNNDrugPredictor(
        task="regression",
        gnn_type=gnn_type,
        hidden_channels=16,
        epochs=3,
        batch_size=4,
        device="cpu",
        validation_split=0.25,
    )
    est.fit(SMILES, y)
    pred = est.predict(SMILES)
    assert pred.shape == (len(SMILES),) and np.isfinite(pred).all()
    emb = est.embed(SMILES[:3])
    assert emb.shape == (3, 16) and np.isfinite(emb).all()
    assert est.n_epochs_ == 3 and len(est.history_["val_loss"]) == 3


@pytest.mark.slow
def test_classification_predict_proba_and_invalid_smiles_fallback():
    y = np.array([1, 0] * (len(SMILES) // 2))
    est = gm.GNNDrugPredictor(task="classification", hidden_channels=16, epochs=2, batch_size=4, device="cpu")
    est.fit(SMILES, y)
    proba = est.predict_proba(["CCO", "C1CC"])  # second is invalid
    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert proba[1, 1] == 0.5  # prior for unparsable input
    labels = est.predict(["CCO", "C1CC"])
    assert set(labels) <= {0.0, 1.0}


@pytest.mark.slow
def test_seed_makes_training_deterministic_on_cpu():
    y = np.arange(len(SMILES), dtype=float)
    kw = dict(task="regression", hidden_channels=8, epochs=2, batch_size=4, device="cpu", random_state=7)
    a = gm.GNNDrugPredictor(**kw).fit(SMILES, y).predict(SMILES)
    b = gm.GNNDrugPredictor(**kw).fit(SMILES, y).predict(SMILES)
    np.testing.assert_allclose(a, b, rtol=1e-5)


@pytest.mark.slow
def test_save_and_load_roundtrip(tmp_path):
    y = np.arange(len(SMILES), dtype=float)
    est = gm.GNNDrugPredictor(task="regression", hidden_channels=8, epochs=2, batch_size=4, device="cpu").fit(
        SMILES, y
    )
    path = tmp_path / "gnn.pt"
    est.save(path)
    loaded = gm.GNNDrugPredictor.load(path, device="cpu")
    np.testing.assert_allclose(loaded.predict(SMILES), est.predict(SMILES), rtol=1e-5)


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="fit"):
        gm.GNNDrugPredictor().predict(["CCO"])
