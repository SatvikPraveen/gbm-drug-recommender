"""
Fingerprint (Tanimoto) similarity.

Morgan fingerprints (radius 2, 2048 bits: ECFP4-like) compared with the Tanimoto
coefficient |A∩B| / |A∪B|. Fast, deterministic and the standard baseline for
"do these two molecules look alike".
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd
from rdkit import DataStructs

from ..config import FINGERPRINT_BITS, FINGERPRINT_RADIUS
from ..feature_extraction import mol_from_smiles, morgan_fingerprint

logger = logging.getLogger(__name__)


def tanimoto_matrix(
    smiles_by_drug: Mapping[str, str], radius: int = FINGERPRINT_RADIUS, n_bits: int = FINGERPRINT_BITS
) -> pd.DataFrame:
    """Symmetric drug x drug Tanimoto matrix; drugs with unparsable SMILES get NaN rows/columns."""
    names = list(smiles_by_drug)
    fps = [morgan_fingerprint(mol_from_smiles(smiles_by_drug[n]), radius, n_bits) for n in names]
    n = len(names)
    sim = np.full((n, n), np.nan)
    for i in range(n):
        if fps[i] is None:
            continue
        sim[i, i] = 1.0
        valid = [(j, fps[j]) for j in range(i + 1, n) if fps[j] is not None]
        if valid:
            row = DataStructs.BulkTanimotoSimilarity(fps[i], [fp for _, fp in valid])
            for (j, _), s in zip(valid, row):
                sim[i, j] = sim[j, i] = s
    return pd.DataFrame(sim, index=names, columns=names)


def pairs_above(matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Long-form list of unordered pairs with similarity >= threshold, highest first."""
    a = matrix.to_numpy()
    iu = np.triu_indices_from(a, k=1)
    mask = a[iu] >= threshold
    out = pd.DataFrame(
        {
            "drug_a": matrix.index[iu[0][mask]],
            "drug_b": matrix.columns[iu[1][mask]],
            "similarity": a[iu][mask],
        }
    )
    return out.sort_values("similarity", ascending=False).reset_index(drop=True)
