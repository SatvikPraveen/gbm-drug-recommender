"""
Maximum-common-substructure (MCS) similarity via ``rdkit.Chem.rdFMCS``.

Similarity is the Tanimoto-style ratio
    |MCS| / (|A| + |B| - |MCS|)
counted in heavy atoms, so identical molecules score 1 and molecules sharing
nothing score 0. Ring atoms only match ring atoms and only complete rings are
matched, which keeps the MCS chemically meaningful (no half-rings).

MCS search is NP-hard; each pair is capped by ``timeout`` seconds and the matrix
is built in parallel. Because cost is O(n²·timeout), the pipeline restricts MCS
to the candidate set rather than all ~500 screened drugs.

The previous implementation used the deprecated ``rdkit.Chem.MCS`` module, which
silently failed and produced an all-zero matrix that was then interpreted in the
README. This module raises loudly if an unexpected fraction of pairs fails.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import rdFMCS

from ..config import MCS_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


def mcs_similarity(smiles_a: str, smiles_b: str, timeout: float = MCS_TIMEOUT_SECONDS) -> float:
    """Heavy-atom MCS Tanimoto between two molecules; NaN if either SMILES is invalid."""
    mol_a, mol_b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return float("nan")
    na, nb = mol_a.GetNumHeavyAtoms(), mol_b.GetNumHeavyAtoms()
    if na == 0 or nb == 0:
        return 0.0
    if smiles_a == smiles_b:
        return 1.0
    params = rdFMCS.MCSParameters()
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.BondTyper = rdFMCS.BondCompare.CompareOrder
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = True
    params.Timeout = max(int(np.ceil(timeout)), 1)
    result = rdFMCS.FindMCS([mol_a, mol_b], params)
    common = result.numAtoms
    return common / (na + nb - common)


def mcs_matrix(
    smiles_by_drug: Mapping[str, str],
    timeout: float = MCS_TIMEOUT_SECONDS,
    n_jobs: int = -1,
    max_failure_fraction: float = 0.05,
) -> pd.DataFrame:
    """Symmetric drug x drug MCS-Tanimoto matrix, computed pairwise in parallel."""
    names = list(smiles_by_drug)
    n = len(names)
    pairs = list(combinations(range(n), 2))
    logger.info("MCS: %d drugs, %d pairs, timeout %.1fs/pair, n_jobs=%s", n, len(pairs), timeout, n_jobs)
    values = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=64)(
        delayed(mcs_similarity)(smiles_by_drug[names[i]], smiles_by_drug[names[j]], timeout) for i, j in pairs
    )
    sim = np.full((n, n), np.nan)
    np.fill_diagonal(sim, 1.0)
    for (i, j), v in zip(pairs, values):
        sim[i, j] = sim[j, i] = v
    frame = pd.DataFrame(sim, index=names, columns=names)

    off = np.array(values, dtype=float)
    nan_frac = float(np.isnan(off).mean()) if len(off) else 0.0
    zero_frac = float(np.mean(off == 0)) if len(off) else 0.0
    if nan_frac > max_failure_fraction:
        raise RuntimeError(
            f"{nan_frac:.1%} of MCS pairs failed (invalid SMILES?); refusing to return a silently broken matrix"
        )
    if len(off) and zero_frac > 0.95:
        raise RuntimeError(
            f"{zero_frac:.1%} of MCS pairs are exactly 0; this is the signature of a broken MCS backend"
        )
    logger.info("MCS done: mean off-diagonal similarity %.3f, %.1f%% zeros", np.nanmean(off), 100 * zero_frac)
    return frame
