import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from gbm_drug import feature_extraction as fe

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
ETHANOL = "CCO"


def test_descriptors_match_known_values():
    d = fe.compute_descriptors(Chem.MolFromSmiles(ASPIRIN))
    assert d["MolWt"] == pytest.approx(180.16, abs=0.01)
    assert d["NumHDonors"] == 1
    assert d["NumHAcceptors"] == 3
    assert d["NumAromaticRings"] == 1
    assert 0 < d["qed"] <= 1


def test_descriptors_nan_for_missing_molecule():
    d = fe.compute_descriptors(None)
    assert set(d) == set(fe.MOLECULAR_DESCRIPTORS)
    assert all(np.isnan(v) for v in d.values())


def test_unknown_descriptor_name_is_an_error():
    with pytest.raises(KeyError):
        fe.compute_descriptors(Chem.MolFromSmiles(ASPIRIN), names=["NotADescriptor"])


def test_lipinski_flags():
    assert fe.lipinski_flags(Chem.MolFromSmiles(ASPIRIN))["lipinski_pass"]
    assert not fe.lipinski_flags(None)["lipinski_pass"]


def test_morgan_fingerprint_shape_and_determinism():
    fp1 = fe.fingerprint_to_array(fe.morgan_fingerprint(Chem.MolFromSmiles(CAFFEINE)))
    fp2 = fe.fingerprint_to_array(fe.morgan_fingerprint(Chem.MolFromSmiles(CAFFEINE)))
    assert fp1.shape == (fe.FINGERPRINT_BITS,)
    assert fp1.dtype == np.uint8
    assert fp1.sum() > 0
    np.testing.assert_array_equal(fp1, fp2)
    assert fe.morgan_fingerprint(None) is None


def test_murcko_scaffold():
    assert fe.murcko_scaffold(Chem.MolFromSmiles(ASPIRIN)) == "c1ccccc1"
    assert fe.murcko_scaffold(Chem.MolFromSmiles(ETHANOL)) == ""  # acyclic
    assert fe.murcko_scaffold(None) == ""


def test_build_feature_table_and_matrices():
    smiles = {"aspirin": ASPIRIN, "caffeine": CAFFEINE, "broken": "C1CC"}
    table = fe.build_feature_table(smiles)
    assert list(table["drug_name"]) == ["aspirin", "caffeine", "broken"]
    assert table.loc[2, fe.MOLECULAR_DESCRIPTORS].isna().all()
    X = fe.descriptor_matrix(table)
    assert X.shape == (3, len(fe.MOLECULAR_DESCRIPTORS))
    assert not np.isnan(X).any()  # median-imputed
    M, names = fe.build_fingerprint_matrix(smiles, n_bits=256)
    assert M.shape == (3, 256)
    assert M[2].sum() == 0 and M[0].sum() > 0
    assert names == list(smiles)


def test_small_molecule_smiles_filters_biologics_and_bad_smiles():
    table = pd.DataFrame(
        {
            "drug_name": ["aspirin", "antibody", "broken", "nosmiles"],
            "smiles": [ASPIRIN, "CC", "C1CC", None],
            "is_small_molecule": [True, False, True, True],
        }
    )
    assert fe.small_molecule_smiles(table) == {"aspirin": ASPIRIN}
