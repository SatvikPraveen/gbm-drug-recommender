"""
Structure-based screening flags for drug pairs.

This is a **coarse, rule-based screen**, not a drug–drug interaction predictor.
It flags pairs whose structures carry reactive or incompatible functional-group
combinations and pairs with extreme lipophilicity mismatch. Pharmacokinetic
interactions (CYP450, transporters) are not modelled: an earlier version shipped a
hard-coded CYP table containing none of the screened drugs, which gave every pair
a "safe to combine" label by construction. If curated PK annotations are
available they can be supplied as a table (drug_name, enzyme, role) and will be
used for substrate–inhibitor flags.

Output vocabulary is deliberately neutral: ``no_flag`` means no rule fired, not
that the combination is safe.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen

logger = logging.getLogger(__name__)

ALERT_LEVELS = {"no_flag": 0, "low": 1, "moderate": 2, "high": 3}

# (name, SMARTS in drug A, SMARTS in drug B, level, description)
STRUCTURAL_RULES: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "acid_base_pair",
        "[CX3,SX4](=O)[OX2H1]",
        "[NX3;H2,H1;!$(NC=O);!$(N-a)]",
        "low",
        "carboxylic/sulfonic acid with an aliphatic primary/secondary amine: possible salt formation in co-formulation",
    ),
    (
        "michael_acceptor_thiol",
        "C=CC(=O)[N,O]",
        "[SX2H1]",
        "moderate",
        "Michael acceptor with a free thiol: covalent adduct formation possible",
    ),
    ("nitro_thiol", "[N+](=O)[O-]", "[SX2H1]", "moderate", "nitro group with a free thiol: redox reactivity"),
    (
        "aldehyde_amine",
        "[CX3H1](=O)[#6]",
        "[NX3;H2]",
        "moderate",
        "aldehyde with a primary amine: imine formation",
    ),
    (
        "platinum_thiol",
        "[Pt]",
        "[SX2H1,SX2H0;!$(S(=O))]",
        "moderate",
        "platinum complex with a thioether/thiol: platinum sequestration",
    ),
)

LOGP_DIFFERENCE_FLAG = 6.0


def _mol(smiles: str | None) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None


def structural_flags(smiles_a: str, smiles_b: str) -> list[dict[str, str]]:
    mol_a, mol_b = _mol(smiles_a), _mol(smiles_b)
    if mol_a is None or mol_b is None:
        return []
    flags = []
    for name, pat_a, pat_b, level, desc in STRUCTURAL_RULES:
        qa, qb = Chem.MolFromSmarts(pat_a), Chem.MolFromSmarts(pat_b)
        if (mol_a.HasSubstructMatch(qa) and mol_b.HasSubstructMatch(qb)) or (
            mol_a.HasSubstructMatch(qb) and mol_b.HasSubstructMatch(qa)
        ):
            flags.append({"rule": name, "level": level, "description": desc})
    return flags


def physchem_flags(
    smiles_a: str, smiles_b: str, logp_diff: float = LOGP_DIFFERENCE_FLAG
) -> list[dict[str, str]]:
    mol_a, mol_b = _mol(smiles_a), _mol(smiles_b)
    if mol_a is None or mol_b is None:
        return []
    diff = abs(Crippen.MolLogP(mol_a) - Crippen.MolLogP(mol_b))
    if diff > logp_diff:
        return [
            {
                "rule": "logp_mismatch",
                "level": "low",
                "description": f"lipophilicity differs by {diff:.1f} log units: co-formulation may be impractical",
            }
        ]
    return []


def pk_flags(drug_a: str, drug_b: str, annotations: pd.DataFrame | None) -> list[dict[str, str]]:
    """Substrate–inhibitor pairs from a curated (drug_name, enzyme, role) table; empty if none supplied."""
    if annotations is None or annotations.empty:
        return []
    ann = annotations.assign(
        drug_name=annotations["drug_name"].str.lower(), role=annotations["role"].str.lower()
    )
    a, b = ann[ann["drug_name"] == drug_a.lower()], ann[ann["drug_name"] == drug_b.lower()]
    flags = []
    for enzyme in set(a["enzyme"]) & set(b["enzyme"]):
        roles_a, roles_b = (
            set(a.loc[a["enzyme"] == enzyme, "role"]),
            set(b.loc[b["enzyme"] == enzyme, "role"]),
        )
        if ("substrate" in roles_a and "inhibitor" in roles_b) or (
            "substrate" in roles_b and "inhibitor" in roles_a
        ):
            flags.append(
                {
                    "rule": f"{enzyme}_substrate_inhibitor",
                    "level": "moderate",
                    "description": f"{enzyme} substrate combined with a {enzyme} inhibitor: exposure of the substrate may rise",
                }
            )
    return flags


def screen_pairs(
    pairs: Iterable[tuple[str, str]], smiles: Mapping[str, str], pk_annotations: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per pair with the highest alert level and every rule that fired."""
    rows = []
    for a, b in pairs:
        flags = (
            structural_flags(smiles.get(a), smiles.get(b))
            + physchem_flags(smiles.get(a), smiles.get(b))
            + pk_flags(a, b, pk_annotations)
        )
        level = max((f["level"] for f in flags), key=ALERT_LEVELS.get, default="no_flag")
        rows.append(
            {
                "drug_a": a,
                "drug_b": b,
                "alert_level": level,
                "n_flags": len(flags),
                "rules": ";".join(f["rule"] for f in flags),
                "details": " | ".join(f["description"] for f in flags),
                "structures_available": smiles.get(a) is not None and smiles.get(b) is not None,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("alert_level", key=lambda s: s.map(ALERT_LEVELS), ascending=False).reset_index(
            drop=True
        )
        logger.info("Screened %d pairs: %s", len(out), out["alert_level"].value_counts().to_dict())
    return out


def summarize(screened: pd.DataFrame) -> dict[str, float]:
    counts = screened["alert_level"].value_counts().to_dict() if not screened.empty else {}
    return {
        "n_pairs": int(len(screened)),
        **{f"n_{k}": int(counts.get(k, 0)) for k in ALERT_LEVELS},
        "n_without_structures": int((~screened["structures_available"]).sum()) if not screened.empty else 0,
        "note": "no_flag means no rule fired; pharmacokinetic interactions are not modelled unless annotations are supplied",
        "mean_flags": float(np.mean(screened["n_flags"])) if not screened.empty else 0.0,
    }
