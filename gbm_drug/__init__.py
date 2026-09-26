"""
gbm_drug: reproducible analysis of GDSC drug-response data for glioblastoma.

Subpackages are imported lazily by callers (``from gbm_drug.similarity import ...``)
so that ``import gbm_drug`` and ``import gbm_drug.config`` stay cheap and never pull
in PyTorch or RDKit as a side effect.
"""

__version__ = "3.0.0"
__author__ = "Satvik Praveen"

__all__ = ["__version__"]
