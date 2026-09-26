"""
Offline pathway enrichment of drug-target genes with an explicit background.

Question asked: *are the targets of GBM-selective drugs enriched for particular
pathways, relative to the targets of every drug in the screen?* Using all
screened drugs' targets as the background matters: oncology screening libraries
are built from kinase inhibitors and DNA-damaging agents, so testing against the
whole genome would "discover" that the library targets cancer pathways.

Gene sets are Enrichr libraries in GMT format, downloaded once to
``data/genesets/`` and then used offline (hypergeometric test, BH-FDR). The
earlier implementation pooled all drugs' targets into one list and posted it to
the Enrichr web API without a background, which is why "Prostate cancer" came
out top.

GDSC target strings are free text ("PI3Kbeta", "CDK4/6", "Broad spectrum kinase
inhibitor"), so :func:`targets_to_genes` maps them to HGNC symbols with a small,
explicit alias table and reports what it could not map.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from .config import (
    ENRICHR_LIBRARY_URL,
    GBM_PATHWAY_KEYWORDS,
    GENESET_DIR,
    GENESET_LIBRARIES,
    PATHWAY_FDR,
    PATHWAY_MIN_OVERLAP,
)

logger = logging.getLogger(__name__)

# GDSC target token -> HGNC symbols. Only unambiguous, well-known aliases; anything not
# listed and not already a plausible symbol is reported as unmapped rather than guessed.
TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "BCR-ABL": ("ABL1",),
    "ABL": ("ABL1",),
    "C-MET": ("MET",),
    "VEGFR": ("KDR", "FLT1", "FLT4"),
    "VEGFR1": ("FLT1",),
    "VEGFR2": ("KDR",),
    "VEGFR3": ("FLT4",),
    "PDGFR": ("PDGFRA", "PDGFRB"),
    "FGFR": ("FGFR1", "FGFR2", "FGFR3"),
    "EGFR": ("EGFR",),
    "HER2": ("ERBB2",),
    "ERBB": ("EGFR", "ERBB2"),
    "MEK": ("MAP2K1", "MAP2K2"),
    "MEK1": ("MAP2K1",),
    "MEK2": ("MAP2K2",),
    "ERK": ("MAPK1", "MAPK3"),
    "ERK1": ("MAPK3",),
    "ERK2": ("MAPK1",),
    "ERK5": ("MAPK7",),
    "JNK": ("MAPK8", "MAPK9"),
    "P38": ("MAPK14",),
    "P38MAPK": ("MAPK14",),
    "PI3K": ("PIK3CA",),
    "PI3KALPHA": ("PIK3CA",),
    "PI3KA": ("PIK3CA",),
    "PI3KBETA": ("PIK3CB",),
    "PI3KB": ("PIK3CB",),
    "PI3KGAMMA": ("PIK3CG",),
    "PI3KDELTA": ("PIK3CD",),
    "PI3KD": ("PIK3CD",),
    "MTOR": ("MTOR",),
    "MTORC1": ("MTOR",),
    "MTORC2": ("MTOR",),
    "AKT": ("AKT1", "AKT2"),
    "RAF": ("RAF1", "BRAF"),
    "CRAF": ("RAF1",),
    "PKC": ("PRKCA",),
    "CDK": ("CDK1", "CDK2"),
    "AURORA": ("AURKA", "AURKB"),
    "AURK": ("AURKA", "AURKB"),
    "HDAC": ("HDAC1", "HDAC2", "HDAC3"),
    "PARP": ("PARP1", "PARP2"),
    "PARP1/2": ("PARP1", "PARP2"),
    "TOP2": ("TOP2A", "TOP2B"),
    "TOPO": ("TOP1",),
    "TUBULIN": ("TUBB",),
    "MICROTUBULE": ("TUBB",),
    "BCL2": ("BCL2",),
    "BCL-2": ("BCL2",),
    "BCL-XL": ("BCL2L1",),
    "MCL1": ("MCL1",),
    "IGF1R": ("IGF1R",),
    "IGFR": ("IGF1R",),
    "IKK": ("CHUK", "IKBKB"),
    "IKKB": ("IKBKB",),
    "TBK1": ("TBK1",),
    "SRC": ("SRC",),
    "P53": ("TP53",),
    "MDM2": ("MDM2",),
    "HSP90": ("HSP90AA1",),
    "PLK": ("PLK1",),
    "PLK1": ("PLK1",),
    "WEE1": ("WEE1",),
    "CHK1": ("CHEK1",),
    "CHK2": ("CHEK2",),
    "ATR": ("ATR",),
    "ATM": ("ATM",),
    "DNAPK": ("PRKDC",),
    "DNA-PK": ("PRKDC",),
    "GSK3": ("GSK3B",),
    "GSK3B": ("GSK3B",),
    "GSK3A": ("GSK3A",),
    "DYRK1B": ("DYRK1B",),
    "KS6B1": ("RPS6KB1",),
    "P70S6K": ("RPS6KB1",),
    "S6K1": ("RPS6KB1",),
    "RSK": ("RPS6KA1",),
    "ROCK": ("ROCK1", "ROCK2"),
    "ROCK1": ("ROCK1",),
    "ROCK2": ("ROCK2",),
    "FLT1": ("FLT1",),
    "C-FGR": ("FGR",),
    "FGR": ("FGR",),
    "FAK": ("PTK2",),
    "PYK2": ("PTK2B",),
    "EPHB4": ("EPHB4",),
    "TRK": ("NTRK1",),
    "TRKA": ("NTRK1",),
    "SMO": ("SMO",),
    "BET": ("BRD4",),
    "BRD2/3/4": ("BRD2", "BRD3", "BRD4"),
    "EZH2": ("EZH2",),
    "XIAP": ("XIAP",),
    "IAP": ("XIAP", "BIRC2"),
    "MCT1": ("SLC16A1",),
    "MCT4": ("SLC16A3",),
    "TNKS1/2": ("TNKS", "TNKS2"),
    "TANK": ("TNKS",),
    "AMPK": ("PRKAA1",),
    "LCK": ("LCK",),
    "FYN": ("FYN",),
    "YES": ("YES1",),
    "BMX": ("BMX",),
    "BTK": ("BTK",),
    "SYK": ("SYK",),
    "JAK": ("JAK1", "JAK2"),
    "STAT3": ("STAT3",),
    "STAT5": ("STAT5A",),
    "ESR1": ("ESR1",),
    "ER": ("ESR1",),
    "AR": ("AR",),
    "GR": ("NR3C1",),
    "PPARG": ("PPARG",),
    "RXR": ("RXRA",),
    "RAR": ("RARA",),
    "THYMIDYLATE SYNTHASE": ("TYMS",),
    "DHFR": ("DHFR",),
    "RRM1": ("RRM1",),
    "RRM2": ("RRM2",),
    "HIF-1A": ("HIF1A",),
    "HIF1A": ("HIF1A",),
    "TERT": ("TERT",),
    "KSP": ("KIF11",),
    "EG5": ("KIF11",),
    "KIF11": ("KIF11",),
    "AURKB": ("AURKB",),
    "AURKC": ("AURKC",),
    "CSNK2A1": ("CSNK2A1",),
    "CK2": ("CSNK2A1",),
    "CDC7": ("CDC7",),
    "TTK": ("TTK",),
    "MPS1": ("TTK",),
    "ULK1": ("ULK1",),
    "VPS34": ("PIK3C3",),
    "PIK3C3": ("PIK3C3",),
    "FEN1": ("FEN1",),
    "TAF1": ("TAF1",),
    "PDK1 (PDPK1)": ("PDPK1",),
    "PDPK1": ("PDPK1",),
    "PDK1": ("PDPK1",),
    "IRAK4": ("IRAK4",),
    "PAK4": ("PAK4",),
    "PAK1": ("PAK1",),
    "NAMPT": ("NAMPT",),
    "IDH1": ("IDH1",),
    "IDH2": ("IDH2",),
    "DOT1L": ("DOT1L",),
    "LSD1": ("KDM1A",),
    "KDM1A": ("KDM1A",),
    "BRD4": ("BRD4",),
    "CBP": ("CREBBP",),
    "P300": ("EP300",),
    "PIM": ("PIM1",),
    "PIM1": ("PIM1",),
    "MAP2K7": ("MAP2K7",),
    "MNK1": ("MKNK1",),
    "MNK2": ("MKNK2",),
    "ERBB4": ("ERBB4",),
    "ERBB2": ("ERBB2",),
    "ERBB3": ("ERBB3",),
    "MET": ("MET",),
    "ALK": ("ALK",),
    "ROS1": ("ROS1",),
    "RET": ("RET",),
    "KIT": ("KIT",),
    "FLT3": ("FLT3",),
    "CSF1R": ("CSF1R",),
    "AXL": ("AXL",),
    "TIE2": ("TEK",),
    "TEK": ("TEK",),
    "DDR1": ("DDR1",),
    "DDR2": ("DDR2",),
    "NTRK1": ("NTRK1",),
    "TNKS": ("TNKS",),
    "PORCN": ("PORCN",),
    "GLI1": ("GLI1",),
    "NOTCH": ("NOTCH1",),
    "GAMMA-SECRETASE": ("PSEN1",),
    "PROTEASOME": ("PSMB5",),
    "PSMB5": ("PSMB5",),
    "HSP70": ("HSPA1A",),
    "NAE": ("NAE1",),
    "MDM4": ("MDM4",),
    "SURVIVIN": ("BIRC5",),
    "BIRC5": ("BIRC5",),
    "USP1": ("USP1",),
    "WNT": ("CTNNB1",),
    "B-CATENIN": ("CTNNB1",),
    "TGFB1": ("TGFB1",),
    "TGFBR1": ("TGFBR1",),
    "ALK5": ("TGFBR1",),
    "SMAD": ("SMAD2", "SMAD3"),
    "MGMT": ("MGMT",),
    "TYMS": ("TYMS",),
    "TOP1": ("TOP1",),
    "TOP2A": ("TOP2A",),
    "TOP2B": ("TOP2B",),
}

_SPLIT = re.compile(r"[,;]|\band\b")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9-]{1,9}$")


def targets_to_genes(text) -> tuple[set[str], set[str]]:
    """
    Map one GDSC PUTATIVE_TARGET string to (HGNC symbols, unmapped tokens).

    "CDK4/6" expands to CDK4 and CDK6; "PI3Kbeta" maps via the alias table; free text
    such as "Broad spectrum kinase inhibitor" is returned as unmapped.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return set(), set()
    genes, unmapped = set(), set()
    for raw in _SPLIT.split(str(text)):
        tok = raw.strip()
        if not tok:
            continue
        key = tok.upper()
        if key in TARGET_ALIASES:
            genes.update(TARGET_ALIASES[key])
            continue
        # "CDK4/6", "JAK1/2", "FGFR1/2/3": stem + alternative suffixes
        m = re.match(r"^([A-Z]+?)(\d+)((?:/\d+)+)$", key)
        if m:
            stem, first, rest = m.groups()
            genes.add(f"{stem}{first}")
            genes.update(f"{stem}{n}" for n in rest.strip("/").split("/"))
            continue
        if "/" in key:  # e.g. "MTORC1/2" handled above; "PI3K/MTOR" split further
            sub_genes, sub_unmapped = set(), set()
            for part in key.split("/"):
                g, u = targets_to_genes(part)
                sub_genes |= g
                sub_unmapped |= u
            genes |= sub_genes
            unmapped |= sub_unmapped
            continue
        if _SYMBOL.match(key) and not key.endswith("-"):
            genes.add(key)
        else:
            unmapped.add(tok)
    return genes, unmapped


def drug_gene_map(summary: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """drug_name -> gene symbols (and -> unmapped tokens) from the drug summary table."""
    genes, unmapped = {}, {}
    for r in summary.itertuples():
        g, u = targets_to_genes(r.putative_target)
        genes[r.drug_name] = g
        if u:
            unmapped[r.drug_name] = u
    n_with = sum(1 for g in genes.values() if g)
    logger.info(
        "Mapped targets for %d/%d drugs; %d drugs have unmapped tokens", n_with, len(genes), len(unmapped)
    )
    return genes, unmapped


# ---------------------------------------------------------------------------
# Gene-set libraries
# ---------------------------------------------------------------------------


def library_path(name: str, directory: Path = GENESET_DIR) -> Path:
    return directory / f"{name}.gmt"


def download_library(name: str, directory: Path = GENESET_DIR, force: bool = False) -> Path:
    """Fetch an Enrichr library as GMT (cached)."""
    path = library_path(name, directory)
    if path.exists() and not force:
        return path
    directory.mkdir(parents=True, exist_ok=True)
    url = ENRICHR_LIBRARY_URL.format(library=name)
    logger.info("Downloading gene-set library %s", name)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.write_text(r.text)
    return path


def load_gmt(path: Path) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        term = parts[0]
        genes = {g.split(",")[0].strip().upper() for g in parts[2:] if g.strip()}
        if genes:
            sets[term] = genes
    return sets


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def enrich(
    query: Iterable[str],
    background: Iterable[str],
    gene_sets: Mapping[str, set[str]],
    min_overlap: int = PATHWAY_MIN_OVERLAP,
) -> pd.DataFrame:
    """
    Hypergeometric over-representation of ``query`` genes vs ``background`` in each gene set.

    Genes outside the background are ignored (in the query and in the gene sets), so
    the universe is exactly the background. Columns: term, overlap, query_size,
    set_size, background_size, p, q (BH), fold_enrichment, genes.
    """
    bg = {g.upper() for g in background}
    q = {g.upper() for g in query} & bg
    N, n = len(bg), len(q)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "term",
                "overlap",
                "query_size",
                "set_size",
                "background_size",
                "p",
                "q",
                "fold_enrichment",
                "genes",
            ]
        )
    rows = []
    for term, genes in gene_sets.items():
        gs = genes & bg
        K = len(gs)
        if K == 0:
            continue
        hit = q & gs
        k = len(hit)
        if k < min_overlap:
            continue
        p = float(hypergeom.sf(k - 1, N, K, n))
        expected = n * K / N
        rows.append(
            {
                "term": term,
                "overlap": k,
                "query_size": n,
                "set_size": K,
                "background_size": N,
                "p": p,
                "fold_enrichment": k / expected if expected else np.nan,
                "genes": ";".join(sorted(hit)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        out["q"] = []
        return out
    out["q"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.sort_values("p").reset_index(drop=True)[
        ["term", "overlap", "query_size", "set_size", "background_size", "p", "q", "fold_enrichment", "genes"]
    ]


def enrich_libraries(
    query: Iterable[str],
    background: Iterable[str],
    libraries: Iterable[str] = GENESET_LIBRARIES,
    directory: Path = GENESET_DIR,
) -> pd.DataFrame:
    """Run :func:`enrich` over several libraries; adds a ``library`` column."""
    frames = []
    for lib in libraries:
        sets = load_gmt(download_library(lib, directory))
        df = enrich(query, background, sets)
        df.insert(0, "library", lib)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def gbm_relevant(
    results: pd.DataFrame, keywords: Iterable[str] = GBM_PATHWAY_KEYWORDS, fdr: float = PATHWAY_FDR
) -> pd.DataFrame:
    if results.empty:
        return results
    pattern = "|".join(re.escape(k) for k in keywords)
    mask = results["term"].str.contains(pattern, case=False, regex=True) & (results["q"] < fdr)
    return results[mask].reset_index(drop=True)
