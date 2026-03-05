"""
Model Comparison Framework

Benchmarks multiple supervised ML models for drug efficacy prediction.

Models:
1. K-Nearest Neighbors (KNN) - Distance-based classification
2. Random Forest (RF) - Ensemble of decision trees
3. Support Vector Machine (SVM) - Maximum margin classifier
4. Neural Network (MLP) - Multi-layer perceptron with hidden layers
5. XGBoost - Gradient boosted trees with regularization

Evaluation:
- Stratified K-Fold Cross-Validation (default: 5 folds)
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Best model selection based on cross-validation score
- Automated hyperparameter tuning

Outputs:
- model_comparison_results.csv - Performance metrics for all models
- Individual model files (.joblib) for each trained classifier
- scaler.joblib - Feature standardization parameters

Usage:
    model_comp = ModelComparison(task_type='classification', random_state=42)
    results = model_comp.train_and_evaluate(X_train, y_train, X_test, y_test)
    model_comp.save_models(output_dir)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import joblib

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Compare multiple ML models for drug response prediction.
    
    Supports both classification (sensitive/resistant) and regression (IC50 prediction).
    """
    
    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        """
        Initialize model comparison framework.
        
        Args:
            task_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize default models based on task type."""
        if self.task_type == 'classification':
            self.models = {
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'SVM': SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=self.random_state
                ),
                'Neural Network': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=self.random_state
                )
            }
            
            if XGBOOST_AVAILABLE:
                self.models['XGBoost'] = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1
                )
        
        else:  # regression
            self.models = {
                'KNN': KNeighborsRegressor(n_neighbors=5),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'SVM': SVR(kernel='rbf'),
                'Neural Network': MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=self.random_state
                )
            }
            
            if XGBOOST_AVAILABLE:
                self.models['XGBoost'] = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1
                )
    
    def train_and_evaluate(self,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          cv_folds: int = 5) -> pd.DataFrame:
        """
        Train all models and evaluate performance.
        
        Args:
            X_train: Training features
            y_train: Training labels/values
            X_test: Test features
            y_test: Test labels/values
            cv_folds: Number of cross-validation folds
            
        Returns:
            DataFrame with model comparison results
        """
        logger.info(f"Training and evaluating {len(self.models)} models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = []
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Cross-validation on training set
                cv_scores = self._cross_validate(model, X_train_scaled, y_train, cv_folds)
                
                # Train on full training set
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                
                # Evaluate
                metrics = self._compute_metrics(y_test, y_pred, model, X_test_scaled)
                
                result = {
                    'Model': model_name,
                    'CV_Score_Mean': cv_scores['mean'],
                    'CV_Score_Std': cv_scores['std'],
                    **metrics
                }
                
                results.append(result)
                self.results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred
                }
                
                logger.info(f"{model_name} - CV Score: {cv_scores['mean']:.4f} ± {cv_scores['std']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        df_results = pd.DataFrame(results)
        
        # Select best model
        if self.task_type == 'classification':
            best_idx = df_results['F1_Score'].idxmax()
        else:
            best_idx = df_results['R2_Score'].idxmax()
        
        self.best_model_name = df_results.loc[best_idx, 'Model']
        self.best_model = self.results[self.best_model_name]['model']
        
        logger.info(f"Best model: {self.best_model_name}")
        
        return df_results
    
    def _cross_validate(self, model, X, y, cv_folds: int) -> Dict[str, Any]:
        """Perform cross-validation."""
        if self.task_type == 'classification':
            scoring = 'f1_weighted'
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            scoring = 'r2'
            cv = cv_folds
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
    
    def _compute_metrics(self, y_true, y_pred, model, X_test) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        
        if self.task_type == 'classification':
            metrics['Accuracy'] = accuracy_score(y_true, y_pred)
            metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['F1_Score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC if binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)[:, 1]
                        metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba)
                except:
                    metrics['ROC_AUC'] = np.nan
        
        else:  # regression
            metrics['MSE'] = mean_squared_error(y_true, y_pred)
            metrics['RMSE'] = np.sqrt(metrics['MSE'])
            metrics['MAE'] = mean_absolute_error(y_true, y_pred)
            metrics['R2_Score'] = r2_score(y_true, y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                metrics['MAPE'] = mape
        
        return metrics
    
    def hyperparameter_tuning(self,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            model_name: str,
                            param_grid: Dict,
                            cv_folds: int = 5) -> Tuple[object, Dict]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of model to tune
            param_grid: Dictionary of parameters to search
            cv_folds: Number of CV folds
            
        Returns:
            Tuple of (best_model, best_params)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        logger.info(f"Hyperparameter tuning for {model_name}...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.task_type == 'classification':
            scoring = 'f1_weighted'
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            scoring = 'r2'
            cv = cv_folds
        
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def get_feature_importance(self, 
                              model_name: str,
                              feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            df_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            df_importance = df_importance.sort_values('Importance', ascending=False)
            df_importance = df_importance.reset_index(drop=True)
            
            return df_importance
        else:
            logger.warning(f"{model_name} does not support feature importance")
            return pd.DataFrame()
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions with trained model.
        
        Args:
            X: Features to predict
            model_name: Name of model to use (uses best model if None)
            
        Returns:
            Predictions
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not trained yet")
        
        X_scaled = self.scaler.transform(X)
        model = self.results[model_name]['model']
        
        return model.predict(X_scaled)
    
    def save_models(self, output_dir: Path):
        """Save all trained models."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        scaler_path = output_dir / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        # Save each model
        for model_name, result in self.results.items():
            model_path = output_dir / f"{model_name.replace(' ', '_').lower()}_model.joblib"
            joblib.dump(result['model'], model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save results summary
        if self.results:
            results_data = []
            for model_name, result in self.results.items():
                results_data.append({
                    'Model': model_name,
                    **result['metrics']
                })
            
            df_results = pd.DataFrame(results_data)
            results_path = output_dir / 'model_comparison_results.csv'
            df_results.to_csv(results_path, index=False)
            logger.info(f"Saved comparison results to {results_path}")
    
    def load_model(self, model_path: Path, scaler_path: Optional[Path] = None):
        """Load a saved model."""
        model = joblib.load(model_path)
        
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
        
        return model


def get_default_param_grids() -> Dict[str, Dict]:
    """
    Get default hyperparameter grids for each model.
    
    Returns:
        Dictionary mapping model names to parameter grids
    """
    param_grids = {
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear', 'poly']
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    }
    
    if XGBOOST_AVAILABLE:
        param_grids['XGBoost'] = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    
    return param_grids
