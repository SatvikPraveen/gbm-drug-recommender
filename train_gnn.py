"""
GNN Training Script for Drug Efficacy Prediction

End-to-end training pipeline for Graph Neural Network model.
Learns directly from SMILES molecular structures to predict drug efficacy.

Workflow:
1. Load drug screening data (IC50 values)
2. Load SMILES representations for drugs
3. Convert SMILES to molecular graphs
4. Train GNN model with cross-validation
5. Evaluate and compare with baseline models
6. Save trained model and predictions

Usage:
    python train_gnn.py --task regression --epochs 100
    python train_gnn.py --task classification --gnn-type gat
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gnn_model import GNNDrugPredictor
from src.config import (
    CLEANED_DATA_FILE, SMILES_DATA_DIR, MODEL_RESULTS_DIR, FIGURES_DIR,
    GNN_HIDDEN_CHANNELS, GNN_NUM_GNN_LAYERS, GNN_NUM_MLP_LAYERS,
    GNN_DROPOUT, GNN_TYPE, GNN_POOLING, GNN_LEARNING_RATE,
    GNN_BATCH_SIZE, GNN_EPOCHS, GNN_EARLY_STOPPING_PATIENCE,
    GNN_VALIDATION_SPLIT, IC50_THRESHOLD_EFFECTIVE, RANDOM_STATE,
    TEST_SIZE, DEVICE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(task='regression'):
    """
    Load drug screening data and SMILES.
    
    Args:
        task: 'regression' for IC50 prediction or 'classification' for effectiveness
        
    Returns:
        Tuple of (smiles_list, targets, drug_names)
    """
    logger.info("Loading drug screening data...")
    
    # Load GDSC data
    gdsc_data = pd.read_csv(CLEANED_DATA_FILE)
    logger.info(f"Loaded {len(gdsc_data)} drug-cell line combinations")
    
    # Load SMILES
    smiles_file = SMILES_DATA_DIR / "drug_smiles.csv"
    smiles_data = pd.read_csv(smiles_file)
    logger.info(f"Loaded SMILES for {len(smiles_data)} drugs")
    
    # Merge data
    merged = gdsc_data.merge(smiles_data, on='drug_name', how='inner')
    logger.info(f"Merged data: {len(merged)} samples")
    
    # Average IC50 across cell lines for each drug
    drug_summary = merged.groupby(['drug_name', 'smiles']).agg({
        'IC50': 'mean',
        'is_effective': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()
    
    logger.info(f"Aggregated to {len(drug_summary)} unique drugs")
    
    # Prepare targets
    if task == 'regression':
        # Log-transform IC50 for better model performance
        targets = np.log1p(drug_summary['IC50'].values)
        logger.info(f"Task: Regression (log-transformed IC50)")
        logger.info(f"Target range: [{targets.min():.2f}, {targets.max():.2f}]")
    else:
        targets = drug_summary['is_effective'].astype(int).values
        logger.info(f"Task: Classification")
        logger.info(f"Class distribution: {np.bincount(targets)}")
    
    smiles_list = drug_summary['smiles'].tolist()
    drug_names = drug_summary['drug_name'].tolist()
    
    return smiles_list, targets, drug_names


def train_gnn_model(args):
    """Train and evaluate GNN model."""
    
    # Load data
    smiles_list, targets, drug_names = load_data(task=args.task)
    
    # Train/test split
    (smiles_train, smiles_test,
     y_train, y_test,
     names_train, names_test) = train_test_split(
        smiles_list, targets, drug_names,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    logger.info(f"Train size: {len(smiles_train)}, Test size: {len(smiles_test)}")
    
    # Initialize GNN model
    logger.info("Initializing GNN model...")
    model = GNNDrugPredictor(
        task=args.task,
        hidden_channels=args.hidden_channels,
        num_gnn_layers=args.num_gnn_layers,
        num_mlp_layers=args.num_mlp_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        pooling=args.pooling,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        random_state=RANDOM_STATE
    )
    
    # Train model
    logger.info("Training GNN model...")
    logger.info(f"Architecture: {args.num_gnn_layers}-layer {args.gnn_type.upper()}, {args.pooling} pooling")
    logger.info(f"Hidden dims: {args.hidden_channels}, Dropout: {args.dropout}")
    
    model.fit(smiles_train, y_train, validation_split=GNN_VALIDATION_SPLIT)
    
    # Evaluate
    logger.info("Evaluating on test set...")
    y_pred = model.predict(smiles_test)
    
    if args.task == 'regression':
        # Convert back from log-space
        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred)
        
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        
        logger.info(f"\nRegression Results:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  R²:   {r2:.4f}")
        
        # Save results
        results_df = pd.DataFrame({
            'drug_name': names_test,
            'true_IC50': y_test_original,
            'predicted_IC50': y_pred_original,
            'error': np.abs(y_test_original - y_pred_original)
        })
        
        metrics = {
            'Model': 'GNN',
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'Task': 'regression'
        }
        
    else:
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"\nClassification Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        
        # If binary classification, calculate ROC AUC
        if len(np.unique(y_test)) == 2:
            y_proba = model.predict_proba(smiles_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            logger.info(f"  ROC AUC:  {roc_auc:.4f}")
        
        # Save results
        results_df = pd.DataFrame({
            'drug_name': names_test,
            'true_label': y_test,
            'predicted_label': y_pred,
            'correct': y_test == y_pred
        })
        
        if len(np.unique(y_test)) == 2:
            results_df['probability'] = y_proba
        
        metrics = {
            'Model': 'GNN',
            'Accuracy': accuracy,
            'ROC_AUC': roc_auc if len(np.unique(y_test)) == 2 else None,
            'Task': 'classification'
        }
    
    # Save results
    output_file = MODEL_RESULTS_DIR / f"gnn_{args.task}_predictions.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved predictions to {output_file}")
    
    # Save metrics
    metrics_file = MODEL_RESULTS_DIR / f"gnn_{args.task}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Save model
    model_file = MODEL_RESULTS_DIR / f"gnn_{args.task}_model.pt"
    model.save(str(model_file))
    logger.info(f"Saved model to {model_file}")
    
    # Plot training history
    plot_training_history(model, args.task)
    
    # Plot predictions
    if args.task == 'regression':
        plot_regression_results(y_test_original, y_pred_original, args.task)
    
    return model, results_df, metrics


def plot_training_history(model, task):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(model.training_history['train_loss']) + 1)
    ax.plot(epochs, model.training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, model.training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'GNN Training History ({task})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = FIGURES_DIR / f"gnn_{task}_training_history.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved training plot to {output_file}")
    plt.close()


def plot_regression_results(y_true, y_pred, task):
    """Plot true vs predicted values for regression."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6, s=100)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('True IC50 (μM)', fontsize=12)
    ax1.set_ylabel('Predicted IC50 (μM)', fontsize=12)
    ax1.set_title('GNN: True vs Predicted IC50', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=100)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted IC50 (μM)', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = FIGURES_DIR / f"gnn_{task}_predictions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved prediction plot to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train GNN model for drug efficacy prediction')
    
    # Task settings
    parser.add_argument('--task', type=str, default='regression',
                       choices=['regression', 'classification'],
                       help='Prediction task type')
    
    # Model architecture
    parser.add_argument('--hidden-channels', type=int, default=GNN_HIDDEN_CHANNELS,
                       help='Hidden dimension size')
    parser.add_argument('--num-gnn-layers', type=int, default=GNN_NUM_GNN_LAYERS,
                       help='Number of GNN layers')
    parser.add_argument('--num-mlp-layers', type=int, default=GNN_NUM_MLP_LAYERS,
                       help='Number of MLP layers')
    parser.add_argument('--dropout', type=float, default=GNN_DROPOUT,
                       help='Dropout probability')
    parser.add_argument('--gnn-type', type=str, default=GNN_TYPE,
                       choices=['gcn', 'gat'],
                       help='Type of GNN layer')
    parser.add_argument('--pooling', type=str, default=GNN_POOLING,
                       choices=['mean', 'max', 'add'],
                       help='Graph pooling method')
    
    # Training settings
    parser.add_argument('--learning-rate', type=float, default=GNN_LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=GNN_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=GNN_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=GNN_EARLY_STOPPING_PATIENCE,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("GNN Drug Efficacy Prediction")
    logger.info("="*80)
    
    # Train model
    model, results, metrics = train_gnn_model(args)
    
    logger.info("\n" + "="*80)
    logger.info("Training complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
