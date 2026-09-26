"""
Task-aware similarity from a trained GNN.

Cosine similarity between graph-level embeddings of a :class:`GNNDrugPredictor`
that was *trained to predict GBM response*. Two drugs are similar here if the
network represents them alike for the purpose of that prediction, which is a
different (and complementary) notion from fingerprint overlap.

An *untrained* GNN also yields embeddings, and cosine similarity of those is a
cheap graph-structure kernel, but it is not "learned" similarity. The earlier
version of this module reported exactly that while describing it as contrastive
learning; ``trained`` in the returned metadata makes the distinction explicit.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..models.gnn_model import GNNDrugPredictor

logger = logging.getLogger(__name__)


def gnn_embeddings(model: GNNDrugPredictor, smiles_by_drug: Mapping[str, str]) -> pd.DataFrame:
    """drug x hidden embedding matrix from the model's encoder (NaN rows for unparsable SMILES)."""
    names = list(smiles_by_drug)
    emb = model.embed([smiles_by_drug[n] for n in names])
    return pd.DataFrame(emb, index=names, columns=[f"h{i}" for i in range(emb.shape[1])])


def gnn_similarity_matrix(model: GNNDrugPredictor, smiles_by_drug: Mapping[str, str]) -> pd.DataFrame:
    """Cosine similarity between GNN embeddings, rescaled from [-1, 1] to [0, 1]."""
    emb = gnn_embeddings(model, smiles_by_drug)
    valid = emb.notna().all(axis=1).to_numpy()
    sim = np.full((len(emb), len(emb)), np.nan)
    if valid.sum():
        cos = cosine_similarity(emb.to_numpy()[valid])
        idx = np.where(valid)[0]
        sim[np.ix_(idx, idx)] = (cos + 1) / 2
    return pd.DataFrame(sim, index=emb.index, columns=emb.index)
