"""
Combination hypotheses from single-agent data.

This module ranks drug *pairs* by a transparent heuristic built from GDSC's own
target and pathway annotations, single-agent GBM potency, and structural
dissimilarity. It is **hypothesis-generating**: no combination screen is used,
so the score is not a synergy prediction and has not been validated against
Bliss or Loewe measurements. :func:`bliss_independence` is provided for when
such measurements exist.

Score components (weights in ``config.COMBINATION_WEIGHTS``):

* ``target_diversity``          1 - overlap coefficient of the two target-gene sets
                                (0.5 when either drug's targets are unannotated)
* ``pathway_complementarity``   1 if GDSC assigns different target pathways, else 0 (0.5 unknown)
* ``potency``                   mean of the two drugs' percentile rank of -z_mean among candidates
* ``structural_novelty``        1 - Tanimoto similarity; pairs above ``COMBINATION_MAX_TANIMOTO``
                                are near-duplicates and are excluded outright

The previous version of this module received no target or pathway data at all,
so every pair scored 0.5 on both, and two EGFR inhibitors (afatinib + gefitinib)
were ranked the #1 "complementary" combination.
"""

from __future__ import annotations

import logging
import re
from itertools import combinations

import numpy as np
import pandas as pd

from .config import COMBINATION_MAX_TANIMOTO, COMBINATION_WEIGHTS

logger = logging.getLogger(__name__)

_SPLIT = re.compile(r"[,;/]|\band\b")


def parse_target_tokens(text) -> set[str]:
    """
    Split a GDSC PUTATIVE_TARGET string into upper-cased tokens.

    Free-text entries such as "Broad spectrum kinase inhibitor" become a single
    token, so two such drugs count as overlapping; unknown / NaN gives an empty set.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return set()
    tokens = {t.strip().upper() for t in _SPLIT.split(str(text)) if t.strip()}
    return {t for t in tokens if t not in {"OTHERS", "OTHER", "UNKNOWN", "NOT DEFINED"}}


def target_diversity(targets_a: set[str], targets_b: set[str]) -> float | None:
    if not targets_a or not targets_b:
        return None
    overlap = len(targets_a & targets_b) / min(len(targets_a), len(targets_b))
    return 1.0 - overlap


def pathway_complementarity(pathway_a, pathway_b) -> float | None:
    if pd.isna(pathway_a) or pd.isna(pathway_b):
        return None
    a, b = str(pathway_a).strip().lower(), str(pathway_b).strip().lower()
    if a in {"other", "unclassified", ""} or b in {"other", "unclassified", ""}:
        return None
    return 0.0 if a == b else 1.0


def score_pairs(
    candidates: pd.DataFrame,
    tanimoto: pd.DataFrame | None = None,
    weights: dict[str, float] = COMBINATION_WEIGHTS,
    max_tanimoto: float = COMBINATION_MAX_TANIMOTO,
) -> pd.DataFrame:
    """
    Score every unordered pair among ``candidates``.

    ``candidates`` needs columns drug_name, putative_target, pathway_name, z_mean and
    (optionally) gbm_selective. Returns one row per pair sorted by total score, with
    every component and a rationale string; excluded near-duplicate pairs are kept
    with ``excluded_reason`` set so the decision is auditable.
    """
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("combination weights must sum to 1")
    cand = candidates.reset_index(drop=True)
    targets = {r.drug_name: parse_target_tokens(r.putative_target) for r in cand.itertuples()}
    pathways = dict(zip(cand["drug_name"], cand["pathway_name"]))
    potency_pct = dict(zip(cand["drug_name"], (-cand["z_mean"]).rank(pct=True)))
    selective = (
        dict(zip(cand["drug_name"], cand["gbm_selective"].astype(bool))) if "gbm_selective" in cand else {}
    )

    rows = []
    for a, b in combinations(cand["drug_name"], 2):
        td = target_diversity(targets[a], targets[b])
        pc = pathway_complementarity(pathways[a], pathways[b])
        sim = (
            float(tanimoto.loc[a, b])
            if tanimoto is not None and a in tanimoto.index and b in tanimoto.columns
            else np.nan
        )
        novelty = 1.0 - sim if not np.isnan(sim) else None
        potency = float((potency_pct[a] + potency_pct[b]) / 2)

        comp = {
            "target_diversity": 0.5 if td is None else td,
            "pathway_complementarity": 0.5 if pc is None else pc,
            "potency": potency,
            "structural_novelty": 0.5 if novelty is None else novelty,
        }
        total = sum(weights[k] * v for k, v in comp.items())

        notes = []
        if td is None:
            notes.append("targets unannotated for at least one drug")
        elif td == 0:
            notes.append("identical target profile")
        elif td >= 0.75:
            notes.append("distinct targets")
        if pc == 1:
            notes.append("different GDSC pathways")
        elif pc == 0:
            notes.append("same GDSC pathway")
        if selective.get(a) and selective.get(b):
            notes.append("both GBM-selective")
        excluded = None
        if not np.isnan(sim) and sim > max_tanimoto:
            excluded = f"near-duplicate structures (Tanimoto {sim:.2f} > {max_tanimoto})"

        rows.append(
            {
                "drug_a": a,
                "drug_b": b,
                "total_score": total if excluded is None else np.nan,
                **comp,
                "tanimoto": sim,
                "targets_a": ";".join(sorted(targets[a])),
                "targets_b": ";".join(sorted(targets[b])),
                "pathway_a": pathways[a],
                "pathway_b": pathways[b],
                "rationale": "; ".join(notes) if notes else "no distinguishing annotation",
                "excluded_reason": excluded,
            }
        )
    out = (
        pd.DataFrame(rows)
        .sort_values(["total_score"], ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    out["rank"] = out["total_score"].rank(ascending=False, method="min").astype("Int64")
    logger.info(
        "Scored %d pairs among %d candidates (%d excluded as near-duplicates)",
        len(out),
        len(cand),
        int(out["excluded_reason"].notna().sum()),
    )
    return out


def pair_matrix(pairs: pd.DataFrame, value: str = "total_score") -> pd.DataFrame:
    """Symmetric drug x drug matrix of a pair-table column."""
    drugs = sorted(set(pairs["drug_a"]) | set(pairs["drug_b"]))
    m = pd.DataFrame(np.nan, index=drugs, columns=drugs)
    for r in pairs.itertuples():
        m.loc[r.drug_a, r.drug_b] = m.loc[r.drug_b, r.drug_a] = getattr(r, value)
    return m


def bliss_independence(
    effect_a: np.ndarray, effect_b: np.ndarray, effect_ab: np.ndarray, tolerance: float = 0.1
) -> dict[str, float | str]:
    """
    Bliss excess for measured single-agent and combination effects (fraction inhibited, 0-1).

    Expected effect under independence is E_A + E_B - E_A*E_B; excess above ``tolerance``
    is called synergistic, below -``tolerance`` antagonistic.
    """
    effect_a, effect_b, effect_ab = (np.asarray(x, float) for x in (effect_a, effect_b, effect_ab))
    expected = effect_a + effect_b - effect_a * effect_b
    excess = effect_ab - expected
    mean_excess = float(np.mean(excess))
    call = (
        "synergistic"
        if mean_excess > tolerance
        else "antagonistic"
        if mean_excess < -tolerance
        else "additive"
    )
    return {
        "bliss_expected": float(np.mean(expected)),
        "bliss_observed": float(np.mean(effect_ab)),
        "bliss_excess": mean_excess,
        "call": call,
    }
