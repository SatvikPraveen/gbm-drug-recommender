"""
End-to-end graph neural network for drug-level GBM response prediction.

``GNNDrugPredictor`` is a scikit-learn-style estimator that takes a list of SMILES
strings, builds molecular graphs with RDKit, and trains a GCN or GAT encoder with
a small MLP head. It plugs into :mod:`gbm_drug.evaluation` on the same folds as
the tabular models, and exposes :meth:`embed` so the trained encoder can be used
as a task-aware molecular similarity (see :mod:`gbm_drug.similarity.gnn_similarity`).

Training details that matter for reproducibility:

* regression targets are standardised internally (fit on the training fold only);
* an internal validation split (``validation_split`` of the *training* data) drives
  early stopping and the learning-rate schedule; the best-validation weights are restored;
* ``random_state`` seeds torch, the internal split and the data loader shuffling;
* BatchNorm requires batches of >1 graph, so the last incomplete training batch is
  dropped when the training set is larger than one batch.

With ~500 molecules this model is small-data deep learning: expect it to be
competitive with, not clearly better than, fingerprint-based gradient boosting.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_add_pool, global_max_pool, global_mean_pool

from ..config import (
    GNN_BATCH_SIZE,
    GNN_DROPOUT,
    GNN_EARLY_STOPPING_PATIENCE,
    GNN_EPOCHS,
    GNN_HIDDEN_CHANNELS,
    GNN_LEARNING_RATE,
    GNN_NUM_GNN_LAYERS,
    GNN_NUM_MLP_LAYERS,
    GNN_POOLING,
    GNN_TYPE,
    GNN_VALIDATION_SPLIT,
    GNN_WEIGHT_DECAY,
    RANDOM_STATE,
    get_device,
)

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


def _guard_openmp(device: torch.device) -> None:
    """
    Avoid a macOS crash when two OpenMP runtimes are loaded in one process.

    scikit-learn and torch each ship their own libomp; if scikit-learn (or XGBoost/umap,
    which use it) is imported before torch, torch's CPU scatter kernels segfault when they
    spin up a thread pool. Pinning torch to one thread sidesteps the clash. Molecular graphs
    are tiny, so CPU threading buys nothing here anyway. Linux builds share one runtime.
    """
    if sys.platform == "darwin" and device.type == "cpu" and torch.get_num_threads() > 1:
        torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Featurisation
# ---------------------------------------------------------------------------

ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "Pt"]
DEGREES = [0, 1, 2, 3, 4, 5, 6]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDISATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
NUM_HS = [0, 1, 2, 3, 4]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _one_hot(value, choices: Sequence) -> list[int]:
    """One-hot with a trailing 'other' slot for values outside ``choices``."""
    vec = [0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else -1] = 1
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    return [
        *_one_hot(atom.GetSymbol(), ATOM_TYPES),
        *_one_hot(atom.GetDegree(), DEGREES),
        *_one_hot(atom.GetFormalCharge(), FORMAL_CHARGES),
        *_one_hot(atom.GetHybridization(), HYBRIDISATIONS),
        *_one_hot(atom.GetTotalNumHs(), NUM_HS),
        *_one_hot(atom.GetChiralTag(), CHIRAL_TAGS),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetMass() / 100.0,
    ]


def bond_features(bond: Chem.Bond) -> list[float]:
    return [*_one_hot(bond.GetBondType(), BOND_TYPES), float(bond.GetIsConjugated()), float(bond.IsInRing())]


NUM_ATOM_FEATURES = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
NUM_BOND_FEATURES = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def smiles_to_graph(smiles: str) -> Data | None:
    """Heavy-atom molecular graph as a PyG ``Data``; None if the SMILES does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        attrs += [feat, feat]
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float)
    else:  # single heavy atom, e.g. some metal salts
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class GNNEncoder(nn.Module):
    """Stack of GCN/GAT layers with BatchNorm, ReLU, dropout, then global pooling."""

    def __init__(
        self, in_channels: int, hidden: int, num_layers: int, dropout: float, gnn_type: str, pooling: str
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden
            if gnn_type == "gcn":
                conv = GCNConv(in_dim, hidden)
            elif gnn_type == "gat":
                heads = 4
                conv = GATConv(in_dim, hidden // heads, heads=heads, concat=True, dropout=dropout)
            else:
                raise ValueError(f"gnn_type must be 'gcn' or 'gat', got {gnn_type!r}")
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden))
        self.pool = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool}[pooling]

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pool(x, batch)


class GNNModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        num_gnn_layers: int,
        num_mlp_layers: int,
        dropout: float,
        gnn_type: str,
        pooling: str,
        out_dim: int,
    ):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden, num_gnn_layers, dropout, gnn_type, pooling)
        layers: list[nn.Module] = []
        dim = hidden
        for _ in range(max(num_mlp_layers - 1, 0)):
            layers += [nn.Linear(dim, dim // 2), nn.BatchNorm1d(dim // 2), nn.ReLU(), nn.Dropout(dropout)]
            dim //= 2
        layers.append(nn.Linear(dim, out_dim))
        self.head = nn.Sequential(*layers)

    def embed(self, data):
        return self.encoder(data.x, data.edge_index, data.batch)

    def forward(self, data):
        return self.head(self.embed(data))


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class GNNDrugPredictor(BaseEstimator):
    """
    scikit-learn-compatible GNN on SMILES.

    Parameters mirror ``gbm_drug.config`` GNN_* settings. ``task`` is "regression"
    (MSE on standardised targets) or "classification" (binary cross-entropy with
    positive-class re-weighting for imbalance).
    """

    def __init__(
        self,
        task: str = "regression",
        hidden_channels: int = GNN_HIDDEN_CHANNELS,
        num_gnn_layers: int = GNN_NUM_GNN_LAYERS,
        num_mlp_layers: int = GNN_NUM_MLP_LAYERS,
        dropout: float = GNN_DROPOUT,
        gnn_type: str = GNN_TYPE,
        pooling: str = GNN_POOLING,
        learning_rate: float = GNN_LEARNING_RATE,
        weight_decay: float = GNN_WEIGHT_DECAY,
        batch_size: int = GNN_BATCH_SIZE,
        epochs: int = GNN_EPOCHS,
        early_stopping_patience: int = GNN_EARLY_STOPPING_PATIENCE,
        validation_split: float = GNN_VALIDATION_SPLIT,
        device: str = "auto",
        random_state: int = RANDOM_STATE,
        verbose: bool = False,
    ):
        self.task = task
        self.hidden_channels = hidden_channels
        self.num_gnn_layers = num_gnn_layers
        self.num_mlp_layers = num_mlp_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.pooling = pooling
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # -- helpers -----------------------------------------------------------

    def _graphs(self, smiles: Sequence[str], y: np.ndarray | None = None) -> tuple[list[Data], np.ndarray]:
        graphs, kept = [], []
        for i, smi in enumerate(smiles):
            g = smiles_to_graph(smi)
            if g is None:
                continue
            if y is not None:
                g.y = torch.tensor([float(y[i])], dtype=torch.float)
            graphs.append(g)
            kept.append(i)
        return graphs, np.asarray(kept, dtype=int)

    def _loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = out.view(-1)
        if self.task == "regression":
            return F.mse_loss(out, y)
        return F.binary_cross_entropy_with_logits(out, y, pos_weight=self._pos_weight)

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model_.train(train)
        total, count = 0.0, 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = batch.to(self.device_)
                if train:
                    self.optimizer_.zero_grad()
                out = self.model_(batch)
                loss = self._loss(out, batch.y.view(-1))
                if train:
                    loss.backward()
                    self.optimizer_.step()
                total += float(loss) * batch.num_graphs
                count += batch.num_graphs
        return total / max(count, 1)

    # -- sklearn API -------------------------------------------------------

    def fit(self, X: Sequence[str], y: np.ndarray):
        if self.task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        torch.manual_seed(self.random_state)
        self.device_ = torch.device(get_device(self.device))
        _guard_openmp(self.device_)

        y = np.asarray(y, dtype=float)
        if self.task == "regression":
            self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1.0)
            y_t = (y - self.y_mean_) / self.y_std_
        else:
            y_t = y
            pos = max(float(y.sum()), 1.0)
            self._pos_weight = torch.tensor([(len(y) - pos) / pos], dtype=torch.float, device=self.device_)

        graphs, kept = self._graphs(list(X), y_t)
        if len(graphs) < 4:
            raise ValueError("need at least 4 valid molecules to train")

        strat = (
            y[kept].astype(int)
            if self.task == "classification" and 0 < y[kept].sum() < len(kept) - 1
            else None
        )
        try:
            train_g, val_g = train_test_split(
                graphs, test_size=self.validation_split, random_state=self.random_state, stratify=strat
            )
        except ValueError:  # too few positives to stratify
            train_g, val_g = train_test_split(
                graphs, test_size=self.validation_split, random_state=self.random_state
            )

        gen = torch.Generator().manual_seed(self.random_state)
        drop_last = len(train_g) > self.batch_size
        train_loader = DataLoader(
            train_g, batch_size=self.batch_size, shuffle=True, drop_last=drop_last, generator=gen
        )
        val_loader = DataLoader(val_g, batch_size=max(self.batch_size, 2), shuffle=False)

        self.model_ = GNNModel(
            NUM_ATOM_FEATURES,
            self.hidden_channels,
            self.num_gnn_layers,
            self.num_mlp_layers,
            self.dropout,
            self.gnn_type,
            self.pooling,
            out_dim=1,
        ).to(self.device_)
        self.optimizer_ = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_, mode="min", factor=0.5, patience=max(self.early_stopping_patience // 2, 1)
        )

        best, best_state, patience = float("inf"), None, 0
        self.history_ = {"train_loss": [], "val_loss": []}
        for epoch in range(self.epochs):
            tr = self._run_epoch(train_loader, train=True)
            va = self._run_epoch(val_loader, train=False)
            scheduler.step(va)
            self.history_["train_loss"].append(tr)
            self.history_["val_loss"].append(va)
            if va < best - 1e-6:
                best, patience = va, 0
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience += 1
            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info("epoch %d train %.4f val %.4f", epoch + 1, tr, va)
            if patience >= self.early_stopping_patience:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_loss_ = best
        self.n_epochs_ = len(self.history_["train_loss"])
        return self

    def _forward_all(self, X: Sequence[str], embed: bool = False) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("call fit() first")
        graphs, kept = self._graphs(list(X))
        width = self.hidden_channels if embed else 1
        out = np.full((len(X), width), np.nan, dtype=float)
        if not graphs:
            return out
        self.model_.eval()
        loader = DataLoader(graphs, batch_size=max(self.batch_size, 2), shuffle=False)
        chunks = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device_)
                chunks.append((self.model_.embed(batch) if embed else self.model_(batch)).cpu().numpy())
        out[kept] = np.vstack(chunks).reshape(len(kept), width)
        return out

    def predict(self, X: Sequence[str]) -> np.ndarray:
        raw = self._forward_all(X)[:, 0]
        if self.task == "regression":
            pred = raw * self.y_std_ + self.y_mean_
        else:
            pred = (raw >= 0).astype(float)
        # Unparsable SMILES fall back to the training-set prior so downstream metrics stay defined.
        fill = self.y_mean_ if self.task == "regression" else 0.0
        return np.where(np.isnan(pred), fill, pred)

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is only defined for classification")
        p = 1 / (1 + np.exp(-self._forward_all(X)[:, 0]))
        p = np.where(np.isnan(p), 0.5, p)
        return np.column_stack([1 - p, p])

    def embed(self, X: Sequence[str]) -> np.ndarray:
        """Graph-level embeddings from the trained encoder (rows of NaN for unparsable SMILES)."""
        return self._forward_all(X, embed=True)

    # -- persistence -------------------------------------------------------

    def save(self, path) -> None:
        torch.save(
            {
                "params": self.get_params(),
                "state": self.model_.state_dict(),
                "y_mean": getattr(self, "y_mean_", None),
                "y_std": getattr(self, "y_std_", None),
                "history": self.history_,
            },
            path,
        )

    @classmethod
    def load(cls, path, device: str = "auto") -> GNNDrugPredictor:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        est = cls(**payload["params"])
        est.device_ = torch.device(get_device(device))
        _guard_openmp(est.device_)
        est.model_ = GNNModel(
            NUM_ATOM_FEATURES,
            est.hidden_channels,
            est.num_gnn_layers,
            est.num_mlp_layers,
            est.dropout,
            est.gnn_type,
            est.pooling,
            out_dim=1,
        ).to(est.device_)
        est.model_.load_state_dict(payload["state"])
        est.y_mean_, est.y_std_ = payload["y_mean"], payload["y_std"]
        est.history_ = payload["history"]
        if est.task == "classification":
            est._pos_weight = torch.tensor([1.0], device=est.device_)
        return est
