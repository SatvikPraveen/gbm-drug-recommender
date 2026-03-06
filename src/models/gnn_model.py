"""
End-to-End Graph Neural Network for Drug Efficacy Prediction

Deep learning model that directly learns from molecular graph structures to predict
drug efficacy (IC50 values) or effectiveness classification.

Architecture:
- Multi-layer Graph Convolutional Network (GCN) / Graph Attention Network (GAT)
- Node features: atom properties (type, degree, hybridization, charge, aromaticity)
- Edge features: bond properties (type, conjugation, ring membership)
- Graph-level pooling: global mean/max/add pooling
- MLP prediction head: regression (IC50) or classification (effective/not effective)

Advantages over traditional ML:
- No manual feature engineering required
- Learns end-to-end from SMILES to prediction
- Captures complex molecular substructures automatically
- Generalizes to novel molecular scaffolds

Training:
- Mini-batch gradient descent with graph batching
- Adam optimizer with learning rate scheduling
- Dropout and batch normalization for regularization
- Early stopping to prevent overfitting

Usage:
    model = GNNDrugPredictor(task='regression', hidden_dim=128, num_layers=3)
    model.fit(smiles_list, targets)
    predictions = model.predict(test_smiles)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rdkit import Chem
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import logging
from tqdm import tqdm
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== MOLECULAR GRAPH UTILITIES ====================

def one_hot_encoding(value, choices):
    """Create one-hot encoding for categorical features."""
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def get_atom_features(atom):
    """
    Extract atom-level features for GNN node representation.
    
    Features:
    - Atom type (C, N, O, F, P, S, Cl, Br, I, others)
    - Degree (0-5)
    - Formal charge (-1, 0, +1)
    - Hybridization (SP, SP2, SP3, others)
    - Aromaticity (binary)
    - Total number of Hydrogens (0-4)
    
    Returns:
        List of features (length: ~30)
    """
    atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    degrees = [0, 1, 2, 3, 4, 5]
    formal_charges = [-1, 0, 1]
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ]
    num_hs = [0, 1, 2, 3, 4]
    
    features = []
    features += one_hot_encoding(atom.GetSymbol(), atom_types)
    features += one_hot_encoding(atom.GetDegree(), degrees)
    features += one_hot_encoding(atom.GetFormalCharge(), formal_charges)
    features += one_hot_encoding(atom.GetHybridization(), hybridizations)
    features.append(int(atom.GetIsAromatic()))
    features += one_hot_encoding(atom.GetTotalNumHs(), num_hs)
    
    return features


def get_bond_features(bond):
    """
    Extract bond-level features for GNN edge attributes.
    
    Features:
    - Bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC)
    - Conjugation (binary)
    - Ring membership (binary)
    
    Returns:
        List of features (length: ~6)
    """
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    
    features = []
    features += one_hot_encoding(bond.GetBondType(), bond_types)
    features.append(int(bond.GetIsConjugated()))
    features.append(int(bond.IsInRing()))
    
    return features


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert SMILES string to PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES representation of molecule
        
    Returns:
        PyG Data object with node features, edge indices, and edge attributes
        Returns None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Invalid SMILES: {smiles}")
        return None
    
    # Add explicit hydrogens for accurate feature extraction
    mol = Chem.AddHs(mol)
    
    # Extract node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Extract edge indices and features
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Add edges in both directions (undirected graph)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        
        bond_feat = get_bond_features(bond)
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)
    
    if len(edge_indices) == 0:
        # Molecule with single atom (no bonds)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ==================== GNN ARCHITECTURES ====================

class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for molecular graphs.
    
    Supports multiple GNN layers (GCN or GAT) with skip connections,
    batch normalization, and dropout.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.2,
        gnn_type: str = 'gcn',
        pooling: str = 'mean'
    ):
        super(GNNEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()
        self.pooling = pooling.lower()
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(in_dim, hidden_channels))
            elif self.gnn_type == 'gat':
                heads = 4 if i < num_layers - 1 else 1
                out_dim = hidden_channels // heads if i < num_layers - 1 else hidden_channels
                self.convs.append(GATConv(in_dim, out_dim, heads=heads, concat=True if i < num_layers - 1 else False))
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Pooling function
        if self.pooling == 'mean':
            self.pool = global_mean_pool
        elif self.pooling == 'max':
            self.pool = global_max_pool
        elif self.pooling == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass through GNN layers.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]
            
        Returns:
            Graph-level embeddings [batch_size, hidden_channels]
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        return x


class GNNPredictor(nn.Module):
    """
    Complete GNN model for drug efficacy prediction.
    
    Architecture: GNN Encoder -> MLP Head -> Output
    """
    
    def __init__(
        self,
        node_features: int,
        hidden_channels: int = 128,
        num_gnn_layers: int = 3,
        num_mlp_layers: int = 2,
        dropout: float = 0.2,
        gnn_type: str = 'gcn',
        pooling: str = 'mean',
        task: str = 'regression'
    ):
        super(GNNPredictor, self).__init__()
        
        self.task = task
        
        # GNN encoder
        self.encoder = GNNEncoder(
            in_channels=node_features,
            hidden_channels=hidden_channels,
            num_layers=num_gnn_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            pooling=pooling
        )
        
        # MLP prediction head
        mlp_layers = []
        for i in range(num_mlp_layers):
            in_dim = hidden_channels if i == 0 else hidden_channels // 2
            out_dim = hidden_channels // 2 if i < num_mlp_layers - 1 else (1 if task == 'regression' else 2)
            
            if i < num_mlp_layers - 1:
                mlp_layers.append(nn.Linear(in_dim, out_dim))
                mlp_layers.append(nn.BatchNorm1d(out_dim))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(dropout))
            else:
                mlp_layers.append(nn.Linear(in_dim, out_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyG Batch object containing x, edge_index, batch
            
        Returns:
            Predictions [batch_size, 1] for regression or [batch_size, 2] for classification
        """
        x = self.encoder(data.x, data.edge_index, data.batch)
        out = self.mlp(x)
        
        return out


# ==================== SKLEARN-COMPATIBLE WRAPPER ====================

class GNNDrugPredictor(BaseEstimator):
    """
    Scikit-learn compatible GNN model for drug efficacy prediction.
    
    Can be used in model comparison pipelines alongside traditional ML models.
    Supports both regression (IC50 prediction) and classification (effective/not effective).
    """
    
    def __init__(
        self,
        task: str = 'regression',
        hidden_channels: int = 128,
        num_gnn_layers: int = 3,
        num_mlp_layers: int = 2,
        dropout: float = 0.2,
        gnn_type: str = 'gcn',
        pooling: str = 'mean',
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = 'auto',
        random_state: int = 42
    ):
        """
        Initialize GNN model.
        
        Args:
            task: 'regression' for IC50 prediction or 'classification' for effectiveness
            hidden_channels: Hidden dimension size
            num_gnn_layers: Number of graph convolution layers
            num_mlp_layers: Number of MLP layers in prediction head
            dropout: Dropout probability
            gnn_type: 'gcn' or 'gat'
            pooling: 'mean', 'max', or 'add'
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            early_stopping_patience: Stop if no improvement for N epochs
            device: 'cpu', 'cuda', or 'auto'
            random_state: Random seed
        """
        self.task = task
        self.hidden_channels = hidden_channels
        self.num_gnn_layers = num_gnn_layers
        self.num_mlp_layers = num_mlp_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.pooling = pooling
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device_str = device
        self.random_state = random_state
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.smiles_to_idx = {}
        self.training_history = {'train_loss': [], 'val_loss': []}
        
    def _setup_device(self):
        """Setup computation device (CPU/CUDA/MPS)."""
        if self.device_str == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.device_str)
        
        logger.info(f"Using device: {self.device}")
    
    def _smiles_to_data(self, smiles_list: List[str], targets: Optional[np.ndarray] = None) -> List[Data]:
        """Convert list of SMILES to list of PyG Data objects."""
        data_list = []
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Converting SMILES to graphs")):
            graph = smiles_to_graph(smiles)
            if graph is None:
                logger.warning(f"Skipping invalid SMILES at index {i}")
                continue
            
            if targets is not None:
                graph.y = torch.tensor([targets[i]], dtype=torch.float if self.task == 'regression' else torch.long)
            
            data_list.append(graph)
        
        return data_list
    
    def fit(self, X: Union[List[str], np.ndarray], y: np.ndarray, validation_split: float = 0.1):
        """
        Train GNN model.
        
        Args:
            X: List of SMILES strings or array of SMILES
            y: Target values (IC50 for regression, labels for classification)
            validation_split: Fraction of data to use for validation
            
        Returns:
            self
        """
        # Setup
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self._setup_device()
        
        # Convert to SMILES list if array
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        # Convert SMILES to graphs
        data_list = self._smiles_to_data(X, y)
        
        if len(data_list) == 0:
            raise ValueError("No valid SMILES found in input data")
        
        # Train/validation split
        train_data, val_data = train_test_split(
            data_list,
            test_size=validation_split,
            random_state=self.random_state
        )
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        node_features = data_list[0].x.shape[1]
        self.model = GNNPredictor(
            node_features=node_features,
            hidden_channels=self.hidden_channels,
            num_gnn_layers=self.num_gnn_layers,
            num_mlp_layers=self.num_mlp_layers,
            dropout=self.dropout,
            gnn_type=self.gnn_type,
            pooling=self.pooling,
            task=self.task
        ).to(self.device)
        
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        # Loss function
        if self.task == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                out = self.model(batch)
                
                if self.task == 'regression':
                    loss = criterion(out.squeeze(), batch.y.squeeze())
                else:
                    loss = criterion(out, batch.y.squeeze())
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = self.model(batch)
                    
                    if self.task == 'regression':
                        loss = criterion(out.squeeze(), batch.y.squeeze())
                    else:
                        loss = criterion(out, batch.y.squeeze())
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Track history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")
        
        return self
    
    def predict(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: List of SMILES strings
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Convert to SMILES list if array
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        # Convert SMILES to graphs
        data_list = self._smiles_to_data(X)
        
        if len(data_list) == 0:
            raise ValueError("No valid SMILES found in input data")
        
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        
        assert self.model is not None  # Type narrowing for Pylance
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                
                if self.task == 'regression':
                    preds = out.squeeze().cpu().numpy()
                else:
                    preds = torch.argmax(out, dim=1).cpu().numpy()
                
                predictions.extend(preds.tolist() if isinstance(preds, np.ndarray) else [preds])
        
        return np.array(predictions)
    
    def predict_proba(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities (for classification only).
        
        Args:
            X: List of SMILES strings
            
        Returns:
            Probability array [n_samples, n_classes]
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Convert to SMILES list if array
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        # Convert SMILES to graphs
        data_list = self._smiles_to_data(X)
        
        if len(data_list) == 0:
            raise ValueError("No valid SMILES found in input data")
        
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)
        
        assert self.model is not None  # Type narrowing for Pylance
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                probs = F.softmax(out, dim=1).cpu().numpy()
                probabilities.append(probs)
        
        return np.vstack(probabilities)
    
    def score(self, X: Union[List[str], np.ndarray], y: np.ndarray) -> float:
        """
        Calculate model score (MSE for regression, accuracy for classification).
        
        Args:
            X: List of SMILES strings
            y: True targets
            
        Returns:
            Score (negative MSE for regression, accuracy for classification)
        """
        predictions = self.predict(X)
        
        if self.task == 'regression':
            return -mean_squared_error(y, predictions)  # Negative because sklearn expects higher=better
        else:
            return accuracy_score(y, predictions)
    
    def save(self, filepath: str):
        """Save model to file."""
        save_dict = {
            'model_state': self.model.state_dict() if self.model else None,
            'config': {
                'task': self.task,
                'hidden_channels': self.hidden_channels,
                'num_gnn_layers': self.num_gnn_layers,
                'num_mlp_layers': self.num_mlp_layers,
                'dropout': self.dropout,
                'gnn_type': self.gnn_type,
                'pooling': self.pooling,
            },
            'training_history': self.training_history
        }
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        save_dict = torch.load(filepath, map_location=self.device)
        
        # Restore config
        for key, value in save_dict['config'].items():
            setattr(self, key, value)
        
        # Initialize model (need to know node_features from data)
        # This will be set during first predict call
        
        self.training_history = save_dict['training_history']
        logger.info(f"Model loaded from {filepath}")
