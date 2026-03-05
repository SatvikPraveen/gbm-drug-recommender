"""
One-Class SVM Drug Predictor

Semi-supervised learning for identifying promising drug candidates.

Approach:
- Trains only on effective drugs (positive examples)
- No negative examples required
- Learns boundary around effective drug feature space
- Predicts whether new drugs fall within this boundary

Algorithm:
- One-Class SVM with RBF kernel
- Hyperparameter tuning via GridSearchCV
- Cross-validation for model robustness
- Decision function provides confidence scores

Outputs:
- drug_predictions.csv - All drugs with decision scores
- one_class_svm_model.pkl - Trained model
- is_promising flag for top candidates

Usage:
    predictor = OneClassDrugPredictor()
    predictor.fit(X_train, feature_names)
    predictions = predictor.identify_promising_drugs(features_df, feature_cols)
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple, Optional
import logging
import joblib

from ..config import (
    SVM_KERNEL, SVM_NU, SVM_GAMMA, CV_FOLDS, CV_SCORING,
    MODEL_RESULTS_DIR, SCALER_TYPE, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OneClassDrugPredictor:
    """One-Class SVM for identifying effective drugs"""
    
    def __init__(self, kernel: str = SVM_KERNEL, nu: float = SVM_NU, gamma: str = SVM_GAMMA):
        """
        Initialize One-Class SVM predictor
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            nu: Upper bound on fraction of outliers
            gamma: Kernel coefficient
        """
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit One-Class SVM on positive examples (effective drugs)
        
        Args:
            X: Feature matrix (positive examples only)
            feature_names: List of feature names
        """
        logger.info(f"Training One-Class SVM with {len(X)} positive examples")
        logger.info(f"Parameters: kernel={self.kernel}, nu={self.nu}, gamma={self.gamma}")
        
        self.feature_names = feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma
        )
        
        self.model.fit(X_scaled)
        
        # Get training predictions
        train_predictions = self.model.predict(X_scaled)
        n_inliers = np.sum(train_predictions == 1)
        n_outliers = np.sum(train_predictions == -1)
        
        logger.info(f"Training complete:")
        logger.info(f"  Inliers: {n_inliers} ({n_inliers/len(X)*100:.1f}%)")
        logger.info(f"  Outliers: {n_outliers} ({n_outliers/len(X)*100:.1f}%)")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict whether drugs are effective
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (1 = effective/inlier, -1 = not effective/outlier)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision function scores (higher = more similar to training data)
        
        Args:
            X: Feature matrix
            
        Returns:
            Decision scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        
        return scores
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for samples
        
        Args:
            X: Feature matrix
            
        Returns:
            Decision scores
        """
        return self.predict_proba(X)
    
    def identify_promising_drugs(self, drug_features_df: pd.DataFrame,
                                 feature_cols: List[str],
                                 drug_name_col: str = 'drug_name',
                                 score_threshold: float = 0.0) -> pd.DataFrame:
        """
        Identify promising drugs from a dataset
        
        Args:
            drug_features_df: DataFrame with drug features
            feature_cols: List of feature columns to use
            drug_name_col: Name of drug name column
            score_threshold: Minimum decision score threshold
            
        Returns:
            DataFrame with predictions and scores
        """
        logger.info(f"Identifying promising drugs from {len(drug_features_df)} candidates")
        
        # Extract features
        X = drug_features_df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        # Get predictions and scores
        predictions = self.predict(X)
        scores = self.predict_proba(X)
        
        # Create results dataframe
        results_df = drug_features_df[[drug_name_col]].copy()
        results_df['prediction'] = predictions
        results_df['decision_score'] = scores
        results_df['is_promising'] = (predictions == 1) & (scores >= score_threshold)
        
        # Sort by decision score (descending)
        results_df = results_df.sort_values('decision_score', ascending=False)
        
        n_promising = results_df['is_promising'].sum()
        logger.info(f"Found {n_promising} promising drugs ({n_promising/len(results_df)*100:.1f}%)")
        
        return results_df
    
    def cross_validate(self, X: np.ndarray, cv: int = CV_FOLDS) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with CV metrics
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        X_scaled = self.scaler.fit_transform(X)
        
        model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        
        # For One-Class SVM, we can compute novelty detection accuracy
        # by treating all training samples as inliers
        scores = []
        
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        
        for train_idx, val_idx in kfold.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_val = X_scaled[val_idx]
            
            model.fit(X_train)
            val_pred = model.predict(X_val)
            
            # Accuracy: percentage of validation samples classified as inliers
            accuracy = np.sum(val_pred == 1) / len(val_pred)
            scores.append(accuracy)
        
        cv_results = {
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'scores': scores
        }
        
        logger.info(f"CV Accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        
        return cv_results
    
    def grid_search(self, X: np.ndarray,
                    param_grid: Optional[Dict] = None) -> Tuple['OneClassDrugPredictor', Dict]:
        """
        Perform grid search for hyperparameter tuning
        
        Args:
            X: Feature matrix
            param_grid: Parameter grid to search
            
        Returns:
            Tuple of (best model, grid search results)
        """
        if param_grid is None:
            param_grid = {
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'nu': [0.01, 0.05, 0.1, 0.2],
                'gamma': ['scale', 'auto']
            }
        
        logger.info("Performing grid search for hyperparameter tuning")
        logger.info(f"Parameter grid: {param_grid}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        base_model = OneClassSVM()
        
        # Custom scoring: percentage of samples classified as inliers
        def custom_scorer(estimator, X):
            predictions = estimator.predict(X)
            return np.sum(predictions == 1) / len(predictions)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            scoring=custom_scorer,
            cv=CV_FOLDS,
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.3f}")
        
        # Update model with best parameters
        self.kernel = grid_search.best_params_['kernel']
        self.nu = grid_search.best_params_['nu']
        self.gamma = grid_search.best_params_['gamma']
        self.model = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        return self, results
    
    def get_feature_importance(self, X: np.ndarray) -> pd.DataFrame:
        """
        Get approximate feature importance based on support vectors
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if self.feature_names is None:
            logger.warning("Feature names not provided")
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            feature_names = self.feature_names
        
        # For RBF kernel, we can look at dual coefficients and support vectors
        if hasattr(self.model, 'support_vectors_'):
            # Use variance of support vectors as proxy for importance
            sv = self.model.support_vectors_
            importance = np.var(sv, axis=0)
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning("Support vectors not available")
            return pd.DataFrame()
    
    def save_model(self, filepath: str = None):
        """
        Save trained model to file
        
        Args:
            filepath: Output file path
        """
        if filepath is None:
            filepath = MODEL_RESULTS_DIR / "one_class_svm_model.pkl"
        
        if self.model is None:
            logger.warning("No model to save")
            return
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'kernel': self.kernel,
            'nu': self.nu,
            'gamma': self.gamma,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = None):
        """
        Load trained model from file
        
        Args:
            filepath: Model file path
        """
        if filepath is None:
            filepath = MODEL_RESULTS_DIR / "one_class_svm_model.pkl"
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.kernel = model_data['kernel']
        self.nu = model_data['nu']
        self.gamma = model_data['gamma']
        self.feature_names = model_data.get('feature_names')
        
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    
    # Generate positive examples (effective drugs)
    n_positive = 50
    n_features = 10
    X_positive = np.random.randn(n_positive, n_features) + 1  # Shifted distribution
    
    # Generate test data (mix of effective and non-effective)
    n_test = 30
    X_test = np.random.randn(n_test, n_features)
    
    # Train model
    predictor = OneClassDrugPredictor()
    predictor.fit(X_positive)
    
    # Make predictions
    predictions = predictor.predict(X_test)
    scores = predictor.predict_proba(X_test)
    
    print(f"\nTest Predictions:")
    print(f"Predicted effective: {np.sum(predictions == 1)}")
    print(f"Predicted not effective: {np.sum(predictions == -1)}")
    print(f"\nDecision scores range: [{scores.min():.3f}, {scores.max():.3f}]")
