"""
Comparing similarity matrices: Mantel test and summary statistics.

Two distance matrices over the same objects are not independent samples, so
ordinary correlation p-values are wrong. The Mantel test permutes object labels
of one matrix and recomputes the correlation to build a null distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import RANDOM_STATE


def upper_triangle(matrix: pd.DataFrame) -> np.ndarray:
    a = matrix.to_numpy(dtype=float)
    return a[np.triu_indices_from(a, k=1)]


def align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = [d for d in a.index if d in b.index]
    return a.loc[common, common], b.loc[common, common]


def mantel_test(
    a: pd.DataFrame,
    b: pd.DataFrame,
    n_permutations: int = 999,
    method: str = "spearman",
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    """
    Mantel correlation between two similarity matrices over the same drugs.

    Returns r, permutation p-value (one-sided, r_perm >= r_obs), n_drugs and n_pairs.
    Pairs with NaN in either matrix are dropped from every correlation.
    """
    a, b = align(a, b)
    if len(a) < 4:
        raise ValueError("need at least 4 shared drugs for a Mantel test")
    corr = stats.spearmanr if method == "spearman" else stats.pearsonr
    A = a.to_numpy(dtype=float)
    B = b.to_numpy(dtype=float)
    iu = np.triu_indices_from(A, k=1)

    def _r(mat_b: np.ndarray) -> float:
        x, y = A[iu], mat_b[iu]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
            return 0.0
        return float(corr(x[ok], y[ok])[0])

    r_obs = _r(B)
    rng = np.random.default_rng(seed)
    n = len(A)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        if _r(B[np.ix_(perm, perm)]) >= r_obs:
            count += 1
    p = (count + 1) / (n_permutations + 1)
    ok = ~(np.isnan(A[iu]) | np.isnan(B[iu]))
    return {
        "r": r_obs,
        "p": p,
        "n_drugs": n,
        "n_pairs": int(ok.sum()),
        "method": method,
        "n_permutations": n_permutations,
    }


def summarize_matrix(matrix: pd.DataFrame, threshold: float) -> dict[str, float]:
    v = upper_triangle(matrix)
    v = v[~np.isnan(v)]
    return {
        "n_drugs": len(matrix),
        "n_pairs": int(len(v)),
        "mean": float(v.mean()) if len(v) else np.nan,
        "median": float(np.median(v)) if len(v) else np.nan,
        "max": float(v.max()) if len(v) else np.nan,
        f"pairs_ge_{threshold}": int((v >= threshold).sum()),
    }
