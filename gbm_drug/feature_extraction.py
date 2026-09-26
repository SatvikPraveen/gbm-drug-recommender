"""
Molecular featurisation from SMILES with RDKit.

Three representations are produced from the curated SMILES table
(``data/smiles/drug_smiles.csv``, built by ``scripts/fetch_smiles.py``):

* physico-chemical descriptors (``MOLECULAR_DESCRIPTORS`` in config) plus
  Lipinski rule-of-five flags,
* Morgan (ECFP-like) bit fingerprints,
* Bemis–Murcko scaffolds, used to build out-of-distribution CV folds.

Nothing here talks to the network. Compounds flagged ``is_small_molecule == False``
(MW > 1500 Da: antibodies, peptides) are excluded by :func:`small_molecule_smiles`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import FINGERPRINT_BITS, FINGERPRINT_RADIUS, MOLECULAR_DESCRIPTORS, SMILES_FILE

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

# Descriptor name -> callable(mol) -> float. Only names listed in config are computed.
_DESCRIPTOR_FUNCTIONS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Crippen.MolLogP,
    "MolMR": Crippen.MolMR,
    "NumHDonors": Lipinski.NumHDonors,
    "NumHAcceptors": Lipinski.NumHAcceptors,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": Lipinski.HeavyAtomCount,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "NumHeteroatoms": Lipinski.NumHeteroatoms,
    "qed": QED.qed,
}

LIPINSKI_COLUMNS = ["lipinski_MW", "lipinski_LogP", "lipinski_HBD", "lipinski_HBA", "lipinski_pass"]


# ---------------------------------------------------------------------------
# SMILES table
# ---------------------------------------------------------------------------


def load_smiles_table(path=SMILES_FILE) -> pd.DataFrame:
    """Read the curated SMILES table written by scripts/fetch_smiles.py."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/fetch_smiles.py`.")
    return pd.read_csv(path, dtype={"pubchem_cid": "Int64"})


def small_molecule_smiles(table: pd.DataFrame | None = None) -> dict[str, str]:
    """drug_name -> SMILES for entries that parse and are small molecules."""
    table = load_smiles_table() if table is None else table
    keep = table["smiles"].notna() & table["is_small_molecule"].astype(bool)
    out = {}
    for name, smi in zip(table.loc[keep, "drug_name"], table.loc[keep, "smiles"]):
        if Chem.MolFromSmiles(smi) is not None:
            out[name] = smi
    dropped = len(table) - len(out)
    if dropped:
        logger.info("Excluded %d entries without a usable small-molecule SMILES", dropped)
    return out


# ---------------------------------------------------------------------------
# Single-molecule features
# ---------------------------------------------------------------------------


def mol_from_smiles(smiles: str | None) -> Chem.Mol | None:
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    return Chem.MolFromSmiles(str(smiles))


def compute_descriptors(
    mol: Chem.Mol | None, names: Iterable[str] = MOLECULAR_DESCRIPTORS
) -> dict[str, float]:
    """Physico-chemical descriptors; NaN for every name if the molecule is missing."""
    names = list(names)
    if mol is None:
        return dict.fromkeys(names, np.nan)
    out = {}
    for name in names:
        fn = _DESCRIPTOR_FUNCTIONS.get(name)
        if fn is None:
            raise KeyError(f"Unknown descriptor {name!r}; add it to _DESCRIPTOR_FUNCTIONS")
        try:
            out[name] = float(fn(mol))
        except Exception:  # RDKit raises for exotic atoms (e.g. metal complexes) in a few descriptors
            out[name] = np.nan
    return out


def lipinski_flags(mol: Chem.Mol | None) -> dict[str, bool]:
    """Rule-of-five components and the usual 'at most one violation' pass flag."""
    if mol is None:
        return dict.fromkeys(LIPINSKI_COLUMNS, False)
    rules = {
        "lipinski_MW": Descriptors.MolWt(mol) <= 500,
        "lipinski_LogP": Crippen.MolLogP(mol) <= 5,
        "lipinski_HBD": Lipinski.NumHDonors(mol) <= 5,
        "lipinski_HBA": Lipinski.NumHAcceptors(mol) <= 10,
    }
    rules["lipinski_pass"] = sum(rules.values()) >= 3
    return rules


_MORGAN_GENERATORS: dict[tuple[int, int], object] = {}


def _morgan_generator(radius: int, n_bits: int):
    key = (radius, n_bits)
    if key not in _MORGAN_GENERATORS:
        _MORGAN_GENERATORS[key] = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return _MORGAN_GENERATORS[key]


def morgan_fingerprint(
    mol: Chem.Mol | None, radius: int = FINGERPRINT_RADIUS, n_bits: int = FINGERPRINT_BITS
):
    """RDKit ExplicitBitVect Morgan fingerprint, or None if the molecule is missing."""
    if mol is None:
        return None
    return _morgan_generator(radius, n_bits).GetFingerprint(mol)


def fingerprint_to_array(fp) -> np.ndarray:
    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def murcko_scaffold(mol: Chem.Mol | None, generic: bool = False) -> str:
    """
    Bemis–Murcko scaffold SMILES. Acyclic molecules (no ring system) return "".

    ``generic=True`` collapses atom types and bond orders so scaffolds group more
    coarsely, which makes scaffold splits stricter.
    """
    if mol is None:
        return ""
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            core = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(core)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def build_feature_table(
    smiles_by_drug: Mapping[str, str], descriptor_names: Iterable[str] = MOLECULAR_DESCRIPTORS
) -> pd.DataFrame:
    """One row per drug: SMILES, descriptors, Lipinski flags, scaffold."""
    names = list(descriptor_names)
    rows = []
    for drug, smi in smiles_by_drug.items():
        mol = mol_from_smiles(smi)
        row = {"drug_name": drug, "smiles": smi}
        row.update(compute_descriptors(mol, names))
        row.update(lipinski_flags(mol))
        row["scaffold"] = murcko_scaffold(mol)
        rows.append(row)
    table = pd.DataFrame(rows, columns=["drug_name", "smiles", *names, *LIPINSKI_COLUMNS, "scaffold"])
    n_bad = int(table[names].isna().all(axis=1).sum())
    if n_bad:
        logger.warning("%d drugs produced no descriptors (unparsable SMILES)", n_bad)
    return table


def build_fingerprint_matrix(
    smiles_by_drug: Mapping[str, str], radius: int = FINGERPRINT_RADIUS, n_bits: int = FINGERPRINT_BITS
) -> tuple[np.ndarray, list[str]]:
    """Dense uint8 matrix of Morgan bits (rows follow the mapping's order); unparsable rows are all-zero."""
    drugs = list(smiles_by_drug)
    matrix = np.zeros((len(drugs), n_bits), dtype=np.uint8)
    for i, drug in enumerate(drugs):
        fp = morgan_fingerprint(mol_from_smiles(smiles_by_drug[drug]), radius, n_bits)
        if fp is not None:
            matrix[i] = fingerprint_to_array(fp)
    return matrix, drugs


def descriptor_matrix(table: pd.DataFrame, names: Iterable[str] = MOLECULAR_DESCRIPTORS) -> np.ndarray:
    """Float matrix of descriptors with NaNs imputed by the column median (caller should scale)."""
    X = np.array(table[list(names)].to_numpy(dtype=float), dtype=float, copy=True)
    medians = np.nanmedian(X, axis=0)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = np.take(medians, nan_idx[1])
    return X
