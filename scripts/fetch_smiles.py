#!/usr/bin/env python
"""
Resolve SMILES for every drug in the GBM summary table from PubChem, with a manual override file.

Resolution order for each drug:
    1. data/smiles/manual_overrides.csv   (drug_name -> pubchem_cid or smiles, or a curated
                                           "no structure" entry for biologics / internal codes)
    2. PubChem name lookup on the GDSC DRUG_NAME (after stripping dose annotations like "(10 uM)")
    3. PubChem name lookup on each comma-separated alias in DRUG_NAME
    4. PubChem name lookup on each GDSC SYNONYM for the drug's DRUG_IDs
    5. Vendor-code spelling variants ("GSK2256098C" -> "GSK2256098", "JNJ38877605" -> "JNJ-38877605")
Every hit is validated with RDKit and annotated with molecular weight, InChIKey, the parent
(largest organic fragment) structure, fragment count, metal presence and a small-molecule
flag (MW <= 1500). The query that resolved each drug is recorded so the table is auditable.

Writes data/smiles/drug_smiles.csv (one row per GDSC drug name) and data/smiles/unresolved.txt
(every summary drug without a structure, with the curation note when there is one).
Re-running only queries drugs not already resolved unless --refresh is given.

PubChem asks for <= 5 requests/second; this script sleeps 0.25 s between calls.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

# Allow running as `python scripts/<name>.py` without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gbm_drug.config import (
    DRUG_SUMMARY_FILE,
    GDSC_COMPOUNDS_FILE,
    MAX_SMALL_MOLECULE_MW,
    SMILES_FILE,
    SMILES_OVERRIDES_FILE,
)

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger("fetch_smiles")

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPS = "SMILES,ConnectivitySMILES,MolecularWeight,MolecularFormula,Title"
PROPS_LEGACY = "IsomericSMILES,CanonicalSMILES,MolecularWeight,MolecularFormula,Title"
SLEEP = 0.25
METALS = {
    "Pt", "Pd", "Au", "Ru", "Fe", "Cu", "Zn", "Co", "Ni", "Mn", "Ga", "As", "Sb", "Bi", "Ti", "V", "Cr",
    "Mo", "W", "Ir", "Os", "Rh", "Ag", "Hg", "Sn", "Pb", "Al", "Mg", "Ca", "Li", "Na", "K",
}  # fmt: skip

_DOSE_RE = re.compile(r"\s*\(\s*[\d.]+\s*(?:u|µ|n|m)?M\s*\)\s*$", re.IGNORECASE)
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
_CODE_RE = re.compile(r"^([A-Za-z]{2,5})[- ]?(\d{3,8})([A-Za-z]{0,2})$")

COLUMNS = [
    "drug_name",
    "smiles",
    "inchikey",
    "smiles_parent",
    "inchikey_parent",
    "pubchem_cid",
    "pubchem_title",
    "molecular_formula",
    "mol_weight",
    "heavy_atoms",
    "n_fragments",
    "has_metal",
    "is_small_molecule",
    "source",
    "query_used",
    "resolved_at",
]


# ---------------------------------------------------------------------------
# PubChem
# ---------------------------------------------------------------------------


def _get(url: str) -> dict | None:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException as exc:
            logger.warning("request error (%s); retrying", exc)
            time.sleep(2 * (attempt + 1))
            continue
        time.sleep(SLEEP)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code in (503, 429):
            time.sleep(5 * (attempt + 1))
            continue
        logger.warning("HTTP %s for %s", r.status_code, url)
        return None
    return None


def _properties(path: str) -> dict | None:
    data = _get(f"{PUG}/compound/{path}/property/{PROPS}/JSON")
    if data is None:
        data = _get(f"{PUG}/compound/{path}/property/{PROPS_LEGACY}/JSON")
    if not data:
        return None
    props = data["PropertyTable"]["Properties"][0]
    smiles = (
        props.get("SMILES")
        or props.get("IsomericSMILES")
        or props.get("ConnectivitySMILES")
        or props.get("CanonicalSMILES")
    )
    if not smiles:
        return None
    return {
        "pubchem_cid": props["CID"],
        "pubchem_title": props.get("Title"),
        "molecular_formula": props.get("MolecularFormula"),
        "smiles": smiles,
    }


def pubchem_by_name(name: str) -> dict | None:
    return _properties(f"name/{requests.utils.quote(name, safe='')}")


def pubchem_by_cid(cid: int) -> dict | None:
    return _properties(f"cid/{int(cid)}")


# ---------------------------------------------------------------------------
# RDKit annotation
# ---------------------------------------------------------------------------


def annotate(record: dict) -> dict | None:
    """Validate SMILES with RDKit and add structural annotations; None if unparsable."""
    mol = Chem.MolFromSmiles(record["smiles"])
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    mw = Descriptors.MolWt(mol)
    has_metal = bool(symbols & METALS)
    # Parent structure: strip counter-ions/solvent for organic salts; metal complexes are
    # multi-fragment by construction (e.g. cisplatin), so they are kept whole.
    if has_metal or len(frags) == 1:
        parent = mol
    else:
        parent = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
    record.update(
        {
            "smiles": Chem.MolToSmiles(mol),  # RDKit canonical form
            "inchikey": inchi.MolToInchiKey(mol),
            "smiles_parent": Chem.MolToSmiles(parent),
            "inchikey_parent": inchi.MolToInchiKey(parent),
            "mol_weight": round(mw, 3),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "n_fragments": len(frags),
            "has_metal": has_metal,
            "is_small_molecule": mw <= MAX_SMALL_MOLECULE_MW,
        }
    )
    return record


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------


def _code_variants(name: str) -> list[str]:
    """Spelling variants of vendor codes: 'GSK2256098C' -> ['GSK2256098', 'GSK-2256098', ...]."""
    m = _CODE_RE.match(name.strip())
    if not m:
        return []
    prefix, digits, _suffix = m.groups()
    variants = [f"{prefix}{digits}", f"{prefix}-{digits}", f"{prefix} {digits}"]
    return [v for v in variants if v.lower() != name.strip().lower()]


def candidate_queries(drug_name: str, synonyms: list[str]) -> list[tuple[str, str]]:
    """Ordered (source, query) pairs to try for one GDSC drug name."""
    out: list[tuple[str, str]] = []
    base = _DOSE_RE.sub("", drug_name).strip()
    out.append(("pubchem_name", base))
    stripped = _PAREN_RE.sub("", base).strip()
    if stripped and stripped != base:
        out.append(("pubchem_name_noparen", stripped))
    for alias in [a.strip() for a in base.split(",") if a.strip()]:
        if alias != base:
            out.append(("pubchem_alias", alias))
    for syn in synonyms:
        syn = syn.strip()
        if syn:
            out.append(("pubchem_synonym", syn))
    for variant in _code_variants(stripped or base):
        out.append(("pubchem_code_variant", variant))
    seen: set[str] = set()
    dedup = []
    for src, q in out:
        if q.lower() not in seen:
            seen.add(q.lower())
            dedup.append((src, q))
    return dedup


def load_synonyms(drug_ids: str) -> list[str]:
    if not GDSC_COMPOUNDS_FILE.exists():
        return []
    comp = pd.read_csv(GDSC_COMPOUNDS_FILE)
    ids = {int(x) for x in str(drug_ids).split(";") if x}
    syn = comp.loc[comp["DRUG_ID"].isin(ids), "SYNONYMS"].dropna().astype(str)
    return [s for row in syn for s in row.split(",")]


def load_overrides() -> pd.DataFrame:
    if SMILES_OVERRIDES_FILE.exists():
        return pd.read_csv(SMILES_OVERRIDES_FILE, dtype=str).fillna("")
    return pd.DataFrame(columns=["drug_name", "pubchem_cid", "smiles", "note"])


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_one(drug_name: str, drug_ids: str, overrides: pd.DataFrame) -> dict | None:
    row = overrides.loc[overrides["drug_name"] == drug_name]
    if len(row):
        row = row.iloc[0]
        rec = None
        if row.get("pubchem_cid"):
            rec = pubchem_by_cid(int(row["pubchem_cid"]))
        elif row.get("smiles"):
            rec = {
                "pubchem_cid": None,
                "pubchem_title": None,
                "molecular_formula": None,
                "smiles": row["smiles"],
            }
        else:
            # Curated "no structure": biologics and internal codes that PubChem name search
            # would otherwise match to an unrelated compound. Do not fall back to PubChem.
            logger.info("%s: no structure by curation (%s)", drug_name, row.get("note", ""))
            return None
        if rec:
            rec = annotate(rec)
            if rec:
                rec.update({"source": "manual_override", "query_used": row.get("note", "")})
                return rec
        logger.warning("override for %s did not resolve; falling back to PubChem", drug_name)

    for source, query in candidate_queries(drug_name, load_synonyms(drug_ids)):
        rec = pubchem_by_name(query)
        if rec is None:
            continue
        rec = annotate(rec)
        if rec is None:
            logger.warning("%s: PubChem SMILES for %r failed RDKit parsing", drug_name, query)
            continue
        rec.update({"source": source, "query_used": query})
        return rec
    return None


def write_unresolved(summary: pd.DataFrame, resolved: pd.DataFrame, overrides: pd.DataFrame) -> int:
    """List every summary drug without a structure, with the curation note when there is one."""
    missing = sorted(set(summary["drug_name"]) - set(resolved["drug_name"]))
    notes = dict(zip(overrides["drug_name"], overrides["note"])) if "note" in overrides else {}
    lines = [f"{n}\t{notes[n]}" if notes.get(n) else n for n in missing]
    (SMILES_FILE.parent / "unresolved.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(missing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--refresh", action="store_true", help="re-query drugs that already have SMILES")
    parser.add_argument("--only", nargs="*", help="restrict to these drug names")
    parser.add_argument(
        "--annotate-only", action="store_true", help="recompute RDKit annotations for cached rows; no network"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    summary = pd.read_csv(DRUG_SUMMARY_FILE)
    overrides = load_overrides()

    if args.annotate_only:
        cached = pd.read_csv(SMILES_FILE, dtype={"pubchem_cid": "Int64"})
        rows = []
        for rec in cached.to_dict("records"):
            ann = annotate(
                {k: rec[k] for k in ("smiles", "pubchem_cid", "pubchem_title", "molecular_formula")}
            )
            if ann is None:
                logger.warning("%s: cached SMILES no longer parses; dropping", rec["drug_name"])
                continue
            rec.update(ann)
            rows.append(rec)
        out = pd.DataFrame(rows).reindex(columns=COLUMNS).sort_values("drug_name")
        out.to_csv(SMILES_FILE, index=False)
        n_missing = write_unresolved(summary, out, overrides)
        logger.info("Re-annotated %d rows; %d drugs without structure", len(out), n_missing)
        return 0

    if SMILES_FILE.exists() and not args.refresh:
        existing = pd.read_csv(SMILES_FILE, dtype={"pubchem_cid": "Int64"})
    else:
        existing = pd.DataFrame(columns=COLUMNS)
    done = set(existing["drug_name"]) if len(existing) else set()

    todo = summary[["drug_name", "drug_ids"]]
    if args.only:
        todo = todo[todo["drug_name"].isin(args.only)]
        done -= set(args.only)
    todo = todo[~todo["drug_name"].isin(done)]
    logger.info("%d drugs to resolve (%d already cached)", len(todo), len(done))

    records = [r for r in existing.to_dict("records") if r["drug_name"] in done]
    for i, (name, ids) in enumerate(todo.itertuples(index=False), 1):
        rec = resolve_one(name, ids, overrides)
        if rec is None:
            logger.warning("[%d/%d] UNRESOLVED %s", i, len(todo), name)
            continue
        rec["drug_name"] = name
        rec["resolved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        records.append(rec)
        logger.info(
            "[%d/%d] %s <- %s (%s) CID %s",
            i,
            len(todo),
            name,
            rec["query_used"],
            rec["source"],
            rec["pubchem_cid"],
        )

    out = pd.DataFrame(records).reindex(columns=COLUMNS).sort_values("drug_name").reset_index(drop=True)
    SMILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SMILES_FILE, index=False)
    n_missing = write_unresolved(summary, out, overrides)
    logger.info(
        "Resolved %d/%d drugs; %d without structure. Wrote %s", len(out), len(summary), n_missing, SMILES_FILE
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
