"""
Tree-Based Models Module for Claims Anomaly Detection

Implements XGBoost and CatBoost as reconstruction-based anomaly detectors
specifically for insurance claims fraud detection.

Author: ML Engineering Team
Date: 2026-01-22
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ClaimsTreeAutoencoder:
    """
    Tree-based autoencoder for claims fraud detection.
    
    Uses XGBoost or CatBoost to reconstruct each feature based on all others,
    with reconstruction errors serving as fraud indicators.
    
    This approach is particularly effective for claims data because:
    - Fast training on tabular data
    - Native categorical support
    - Feature importance for fraud investigation
    - CPU-friendly deployment
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        config: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize tree-based autoencoder for claims.
        
        Args:
            model_type: "xgboost" or "catboost"
            config: Configuration object from ConfigManager
            **kwargs: Additional model parameters
        """
        if model_type not in ["xgboost", "catboost"]:
            raise ValueError(f"model_type must be 'xgboost' or 'catboost', got '{model_type}'")
        
        if model_type == "xgboost" and not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost>=2.0.0"
            )
        
        if model_type == "catboost" and not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost not installed. Install with: pip install catboost>=1.2.0"
            )
        
        self.model_type = model_type
        self.config = config
        self.kwargs = kwargs
        
        # Model storage: one model per feature
        self.models: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []
        self.feature_types: Dict[str, str] = {}
        
        # Extract configuration
        self._extract_config()
        
        logger.info(f"Initialized ClaimsTreeAutoencoder (type={model_type})")
    
    def _extract_config(self):
        """Extract model configuration from config object or kwargs."""
        if self.config is not None:
            # Try to get tree model config
            try:
                tree_config = self.config.get('tree_models', {})
                if self.model_type == "xgboost":
                    self.model_params = tree_config.get('xgboost', {})
                elif self.model_type == "catboost":
                    self.model_params = tree_config.get('catboost', {})
            except:
                # Fallback to default params
                self.model_params = {}
        else:
            self.model_params = {}
        
        # Default parameters
        if self.model_type == "xgboost":
            default_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            for key, value in default_params.items():
                if key not in self.model_params:
                    self.model_params[key] = value
        
        elif self.model_type == "catboost":
            default_params = {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'random_seed': 42,
                'verbose': False
            }
            for key, value in default_params.items():
                if key not in self.model_params:
                    self.model_params[key] = value
        
        # Override with kwargs
        self.model_params.update(self.kwargs)
    
    def fit(
        self,
        X: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        Fit tree models for fraud detection.
        
        Args:
            X: Claims data DataFrame
            categorical_features: Categorical column names
            numerical_features: Numerical column names
            verbose: Whether to log progress
        
        Returns:
            self
        """
        self.feature_names = list(X.columns)
        
        # Determine feature types
        if categorical_features is not None:
            self.categorical_features = categorical_features
        if numerical_features is not None:
            self.numerical_features = numerical_features
        
        # Build feature type mapping
        for feat in self.feature_names:
            if feat in self.categorical_features:
                self.feature_types[feat] = 'categorical'
            else:
                self.feature_types[feat] = 'numerical'
        
        if verbose:
            logger.info(f"Training fraud detection models for {len(self.feature_names)} features...")
            logger.info(f"  Categorical: {len(self.categorical_features)} features")
            logger.info(f"  Numerical: {len(self.numerical_features)} features")
        
        # Train one model per feature
        for idx, target_feature in enumerate(self.feature_names, 1):
            # All other features as predictors
            predictor_features = [f for f in self.feature_names if f != target_feature]
            
            X_train = X[predictor_features]
            y_train = X[target_feature]
            
            # Identify categorical predictors
            cat_predictors = [f for f in predictor_features if f in self.categorical_features]
            
            # Train model
            if self.model_type == "xgboost":
                model = self._train_xgboost(
                    X_train, y_train, cat_predictors, target_feature
                )
            elif self.model_type == "catboost":
                model = self._train_catboost(
                    X_train, y_train, cat_predictors, target_feature
                )
            
            self.models[target_feature] = model
            
            # Log progress
            if verbose and (idx % 3 == 0 or idx == len(self.feature_names)):
                logger.info(f"  Progress: {idx}/{len(self.feature_names)} models trained")
        
        if verbose:
            logger.info("✓ All fraud detection models trained successfully")
        
        return self
    
    def _train_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_features: List[str],
        target_name: str
    ):
        """Train XGBoost model for a single feature."""
        is_categorical = target_name in self.categorical_features
        
        # Filter out rows with NaN in target variable
        valid_mask = ~y.isna()
        if not valid_mask.all():
            n_nan = (~valid_mask).sum()
            if n_nan > len(y) * 0.5:  # More than 50% NaN
                logger.warning(f"  Feature '{target_name}': {n_nan}/{len(y)} NaN values (>{n_nan/len(y)*100:.1f}%)")
            X = X[valid_mask].copy()
            y = y[valid_mask].copy()
        
        # Convert categorical columns to numeric codes (compatible with all XGBoost versions)
        X_cat = X.copy()
        for col in cat_features:
            if col in X_cat.columns:
                # Convert to numeric codes, replacing -1 (unknown) with 0
                codes = pd.Categorical(X_cat[col]).codes
                # Replace -1 (unknown categories) with a valid code
                codes = np.where(codes == -1, 0, codes)
                X_cat[col] = codes
        
        # Also handle NaN values in predictor features
        # Replace NaN with median for numeric, mode for categorical
        for col in X_cat.columns:
            if X_cat[col].isna().any():
                if col in cat_features:
                    # For categorical, use mode (most frequent value)
                    mode_val = X_cat[col].mode()[0] if not X_cat[col].mode().empty else 0
                    X_cat[col] = X_cat[col].fillna(mode_val)
                else:
                    # For numerical, use median
                    median_val = X_cat[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    X_cat[col] = X_cat[col].fillna(median_val)
        
        # Determine objective and create model
        if is_categorical:
            n_classes = y.nunique()
            # Convert target to numeric if categorical
            y_codes = pd.Categorical(y).codes
            # Replace -1 with 0 for target as well
            y = np.where(y_codes == -1, 0, y_codes)
            
            if n_classes == 2:
                objective = "binary:logistic"
                params = {**self.model_params, 'objective': objective}
                params.pop('num_class', None)
                model = xgb.XGBClassifier(**params)
            else:
                objective = "multi:softprob"
                params = {
                    **self.model_params,
                    'objective': objective,
                    'num_class': n_classes
                }
                model = xgb.XGBClassifier(**params)
        else:
            # Regression - ensure no NaN, inf values
            y = np.array(y, dtype=np.float64)
            if np.isnan(y).any() or np.isinf(y).any():
                logger.error(f"  Feature '{target_name}': Target still contains NaN/inf after cleaning")
                # Replace any remaining NaN/inf with median
                valid_y = y[np.isfinite(y)]
                if len(valid_y) > 0:
                    median_val = np.median(valid_y)
                    y = np.where(np.isfinite(y), y, median_val)
                else:
                    # Fallback to zero if all values are invalid
                    y = np.zeros_like(y)
            
            params = {**self.model_params, 'objective': 'reg:squarederror'}
            params.pop('num_class', None)
            model = xgb.XGBRegressor(**params)
        
        # Train model (no enable_categorical needed)
        model.fit(X_cat, y, verbose=False)
        
        return model
    
    def _train_catboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_features: List[str],
        target_name: str
    ):
        """Train CatBoost model for a single feature."""
        is_categorical = target_name in self.categorical_features
        
        # Filter out rows with NaN in target variable
        valid_mask = ~y.isna()
        if not valid_mask.all():
            n_nan = (~valid_mask).sum()
            if n_nan > len(y) * 0.5:  # More than 50% NaN
                logger.warning(f"  Feature '{target_name}': {n_nan}/{len(y)} NaN values (>{n_nan/len(y)*100:.1f}%)")
            X = X[valid_mask].copy()
            y = y[valid_mask].copy()
        
        # Get indices of categorical features
        cat_feature_indices = [
            X.columns.get_loc(f) for f in cat_features if f in X.columns
        ]
        
        # Create and train model
        if is_categorical:
            model = cb.CatBoostClassifier(
                **self.model_params,
                cat_features=cat_feature_indices
            )
        else:
            model = cb.CatBoostRegressor(
                **self.model_params,
                cat_features=cat_feature_indices
            )
        
        model.fit(X, y, verbose=False)
        
        return model
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict reconstructions for fraud detection.
        
        Args:
            X: Claims data DataFrame
            
        Returns:
            Dictionary mapping feature names to predictions
        """
        if not self.models:
            raise ValueError("Model not trained. Call fit() first.")
        
        predictions = {}
        
        for target_feature in self.feature_names:
            predictor_features = [f for f in self.feature_names if f != target_feature]
            X_pred = X[predictor_features].copy()
            
            # Get categorical predictors for this subset
            cat_predictors = [
                f for f in predictor_features if f in self.categorical_features
            ]
            
            # Handle categorical conversion for XGBoost
            if self.model_type == "xgboost":
                # Convert to numeric codes (same as training)
                for col in cat_predictors:
                    if col in X_pred.columns:
                        codes = pd.Categorical(X_pred[col]).codes
                        # Replace -1 (unknown) with 0
                        X_pred[col] = np.where(codes == -1, 0, codes)
            
            # IMPORTANT: Handle NaN in predictor features (same as training)
            for col in X_pred.columns:
                if X_pred[col].isna().any():
                    if col in cat_predictors:
                        # For categorical, use mode (most frequent value)
                        mode_val = X_pred[col].mode()[0] if not X_pred[col].mode().empty else 0
                        X_pred[col] = X_pred[col].fillna(mode_val)
                    else:
                        # For numerical, use median
                        median_val = X_pred[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0
                        X_pred[col] = X_pred[col].fillna(median_val)
            
            # Make prediction
            model = self.models[target_feature]
            pred = model.predict(X_pred)
            predictions[target_feature] = pred
        
        return predictions
    
    def compute_fraud_scores(
        self,
        X: pd.DataFrame,
        method: str = "l2"
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute fraud scores based on reconstruction error.
        
        Higher scores indicate higher likelihood of fraud.
        
        Args:
            X: Claims data DataFrame
            method: Aggregation method - "l2", "l1", or "max"
            
        Returns:
            Tuple of (aggregate_fraud_scores, per_feature_errors)
        """
        if not self.models:
            raise ValueError("Model not trained. Call fit() first.")
        
        predictions = self.predict(X)
        per_feature_errors = {}
        
        # Compute per-feature reconstruction errors
        for feature_name in self.feature_names:
            actual = X[feature_name].values
            predicted = predictions[feature_name]
            
            if feature_name in self.categorical_features:
                # Classification error (0 = correct, 1 = incorrect)
                # Handle NaN in actual values - treat as always incorrect
                error = np.where(pd.isna(actual), 1.0, (actual != predicted).astype(float))
            else:
                # Regression error (squared difference)
                # Handle NaN in actual values - use 0 error (no information)
                error = np.where(pd.isna(actual), 0.0, (actual - predicted) ** 2)
            
            per_feature_errors[feature_name] = error
        
        # Aggregate errors across features
        all_errors = np.stack(list(per_feature_errors.values()), axis=1)
        
        # Handle any remaining NaN (should not happen now, but safety check)
        all_errors = np.nan_to_num(all_errors, nan=0.0, posinf=0.0, neginf=0.0)
        
        if method == "l2":
            # L2 norm (Euclidean distance)
            aggregate_scores = np.linalg.norm(all_errors, axis=1)
        elif method == "l1":
            # L1 norm (Manhattan distance)
            aggregate_scores = np.sum(np.abs(all_errors), axis=1)
        elif method == "max":
            # Maximum error across features
            aggregate_scores = np.max(all_errors, axis=1)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'l2', 'l1', or 'max'")
        
        return aggregate_scores, per_feature_errors
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importances for fraud investigation.
        
        Returns:
            Nested dict: {target_feature: {predictor: importance}}
        """
        if not self.models:
            raise ValueError("Model not trained. Call fit() first.")
        
        importances = {}
        
        for target_feature, model in self.models.items():
            predictor_features = [f for f in self.feature_names if f != target_feature]
            
            if self.model_type == "xgboost":
                feat_imp = model.feature_importances_
            elif self.model_type == "catboost":
                feat_imp = model.feature_importances_
            
            importances[target_feature] = dict(zip(predictor_features, feat_imp))
        
        return importances
    
    def get_top_fraud_indicators(
        self,
        target_feature: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top K fraud indicators for a specific feature.
        
        Useful for understanding what features most predict fraud patterns.
        
        Args:
            target_feature: Feature to analyze
            top_k: Number of top predictors
            
        Returns:
            List of (feature_name, importance) tuples
        """
        if target_feature not in self.models:
            raise ValueError(f"Unknown feature: {target_feature}")
        
        importances = self.get_feature_importance()[target_feature]
        sorted_features = sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_features[:top_k]
    
    def save(self, path: str):
        """
        Save fraud detection model.
        
        Args:
            path: Path to save directory
        """
        if not self.models:
            raise ValueError("Model not trained. Call fit() first.")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'feature_types': self.feature_types,
            'model_params': self.model_params
        }
        
        with open(save_path / 'config.pkl', 'wb') as f:
            pickle.dump(config_dict, f)
        
        # Save individual models
        for feature_name, model in self.models.items():
            model_file = save_path / f"model_{feature_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Fraud detection model saved to {save_path}")
    
    @classmethod
    def load(cls, path: str):
        """
        Load fraud detection model.
        
        Args:
            path: Path to saved model directory
            
        Returns:
            Loaded ClaimsTreeAutoencoder instance
        """
        load_path = Path(path)
        
        # Load configuration
        with open(load_path / 'config.pkl', 'rb') as f:
            config_dict = pickle.load(f)
        
        # Create instance
        instance = cls(
            model_type=config_dict['model_type'],
            **config_dict['model_params']
        )
        
        # Restore attributes
        instance.feature_names = config_dict['feature_names']
        instance.categorical_features = config_dict['categorical_features']
        instance.numerical_features = config_dict['numerical_features']
        instance.feature_types = config_dict['feature_types']
        
        # Load individual models
        for feature_name in instance.feature_names:
            model_file = load_path / f"model_{feature_name}.pkl"
            with open(model_file, 'rb') as f:
                instance.models[feature_name] = pickle.load(f)
        
        logger.info(f"Fraud detection model loaded from {load_path}")
        return instance
    
    def __repr__(self) -> str:
        """String representation."""
        trained = "trained" if self.models else "untrained"
        n_features = len(self.feature_names) if self.feature_names else 0
        return (
            f"ClaimsTreeAutoencoder(type={self.model_type}, "
            f"n_features={n_features}, status={trained})"
        )


def create_ensemble_fraud_scores(
    models: Dict[str, ClaimsTreeAutoencoder],
    X: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    method: str = "l2"
) -> np.ndarray:
    """
    Create ensemble fraud scores from multiple tree models.
    
    Args:
        models: Dictionary of {model_name: ClaimsTreeAutoencoder}
        X: Claims data
        weights: Optional weights for each model
        method: Scoring method to use
        
    Returns:
        Weighted ensemble fraud scores
    """
    if not models:
        raise ValueError("No models provided")
    
    # Default to equal weights
    if weights is None:
        weights = {name: 1.0 / len(models) for name in models.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {name: w / total_weight for name, w in weights.items()}
    
    # Compute weighted ensemble scores
    ensemble_scores = None
    
    for model_name, model in models.items():
        scores, _ = model.compute_fraud_scores(X, method=method)
        weight = weights.get(model_name, 0.0)
        
        if ensemble_scores is None:
            ensemble_scores = weight * scores
        else:
            ensemble_scores += weight * scores
    
    return ensemble_scores
