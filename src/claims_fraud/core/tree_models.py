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
        # Store category mappings for consistent encoding/decoding
        self.category_mappings: Dict[str, List[str]] = {}  # {feature: [categories]}
        
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
        
        # Build feature type mapping and category mappings
        for feat in self.feature_names:
            if feat in self.categorical_features:
                self.feature_types[feat] = 'categorical'
                # Store unique categories for this feature
                self.category_mappings[feat] = sorted(X[feat].dropna().unique().astype(str).tolist())
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
        """Train XGBoost model for a single feature. XGBoost requires ALL columns numeric."""
        is_categorical = target_name in self.categorical_features
        
        # Filter out rows with NaN in target variable
        valid_mask = ~y.isna()
        if not valid_mask.all():
            n_nan = (~valid_mask).sum()
            if n_nan > len(y) * 0.5:  # More than 50% NaN
                logger.warning(f"  Feature '{target_name}': {n_nan}/{len(y)} NaN values (>{n_nan/len(y)*100:.1f}%)")
            X = X[valid_mask].copy()
            y = y[valid_mask].copy()
        
        # For XGBoost: Convert ALL non-numeric columns to numeric BEFORE handling NaN
        # (because categorical dtype doesn't support median operation)
        X_xgb = X.copy()
        for col in X_xgb.columns:
            if not pd.api.types.is_numeric_dtype(X_xgb[col]):
                # Convert to numeric codes
                X_xgb[col] = pd.Categorical(X_xgb[col]).codes
                # Replace -1 (NaN) with 0
                X_xgb[col] = X_xgb[col].replace(-1, 0).astype(np.int32)
        
        # Handle NaN values in numeric columns
        for col in X_xgb.columns:
            if X_xgb[col].isna().any():
                median_val = X_xgb[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X_xgb[col] = X_xgb[col].fillna(median_val)
        
        # Handle target variable
        if is_categorical:
            n_classes = y.nunique()
            y_codes = pd.Categorical(y).codes
            y = np.where(y_codes == -1, 0, y_codes)
            
            if n_classes == 2:
                objective = "binary:logistic"
                params = {**self.model_params, 'objective': objective}
                params.pop('num_class', None)
                model = xgb.XGBClassifier(**params)
            else:
                objective = "multi:softprob"
                params = {**self.model_params, 'objective': objective, 'num_class': n_classes}
                model = xgb.XGBClassifier(**params)
        else:
            # Check if target is numeric or categorical
            if not pd.api.types.is_numeric_dtype(y):
                # Auto-detect non-numeric target - add to categorical list
                if target_name not in self.categorical_features:
                    self.categorical_features.append(target_name)
                    self.feature_types[target_name] = 'categorical'
                    # Store categories BEFORE encoding
                    self.category_mappings[target_name] = sorted(y.dropna().unique().astype(str).tolist())
                
                y_codes = pd.Categorical(y).codes
                y = np.where(y_codes == -1, 0, y_codes)
                n_classes = len(pd.Categorical(y).categories)
                
                if n_classes == 2:
                    objective = "binary:logistic"
                    params = {**self.model_params, 'objective': objective}
                    params.pop('num_class', None)
                    model = xgb.XGBClassifier(**params)
                else:
                    objective = "multi:softprob"
                    params = {**self.model_params, 'objective': objective, 'num_class': n_classes}
                    model = xgb.XGBClassifier(**params)
            else:
                # Numeric target - regression
                y = np.array(y, dtype=np.float64)
                if np.isnan(y).any() or np.isinf(y).any():
                    valid_y = y[np.isfinite(y)]
                    if len(valid_y) > 0:
                        median_val = np.median(valid_y)
                        y = np.where(np.isfinite(y), y, median_val)
                    else:
                        y = np.zeros_like(y)
                
                params = {**self.model_params, 'objective': 'reg:squarederror'}
                params.pop('num_class', None)
                model = xgb.XGBRegressor(**params)
        
        # Train XGBoost model
        model.fit(X_xgb, y, verbose=False)
        
        return model
    
    def _train_catboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_features: List[str],
        target_name: str
    ):
        """Train CatBoost model for a single feature. CatBoost handles strings natively."""
        is_categorical = target_name in self.categorical_features
        
        # Filter out rows with NaN in target variable
        valid_mask = ~y.isna()
        if not valid_mask.all():
            n_nan = (~valid_mask).sum()
            if n_nan > len(y) * 0.5:  # More than 50% NaN
                logger.warning(f"  Feature '{target_name}': {n_nan}/{len(y)} NaN values (>{n_nan/len(y)*100:.1f}%)")
            X = X[valid_mask].copy()
            y = y[valid_mask].copy()
        
        # CRITICAL FIX: Detect if target contains string/object values even if not marked as categorical
        # This handles cases where the target has values like "30-45" that CatBoost can't parse as floats
        if not is_categorical and not pd.api.types.is_numeric_dtype(y):
            # Target contains non-numeric values but wasn't marked as categorical
            # Convert to categorical for classification
            is_categorical = True
            # IMPORTANT: Add to permanent categorical list so compute_fraud_scores knows
            if target_name not in self.categorical_features:
                self.categorical_features.append(target_name)
                self.feature_types[target_name] = 'categorical'
                # Store categories for this feature
                self.category_mappings[target_name] = sorted(y.dropna().unique().astype(str).tolist())
            logger.info(f"  Auto-detected feature '{target_name}' as categorical (contains non-numeric values)")
        
        # For CatBoost: Keep strings native, just handle NaN and categorical dtype
        X_cat = X.copy()
        
        # CRITICAL: Auto-detect columns with non-numeric values that aren't marked as categorical
        # Add them to the categorical features list for CatBoost
        actual_cat_features = list(cat_features)  # Start with declared categorical features
        
        for col in X_cat.columns:
            if col not in actual_cat_features:  # Only check non-declared categorical columns
                # Check if column contains non-numeric values
                if not pd.api.types.is_numeric_dtype(X_cat[col]):
                    # This column has string/object values but wasn't marked as categorical
                    actual_cat_features.append(col)
                    logger.info(f"  Auto-detected predictor '{col}' as categorical (contains non-numeric values)")
        
        # Convert categorical dtype to string for ALL categorical features (declared + auto-detected)
        for col in actual_cat_features:
            if col in X_cat.columns:
                if pd.api.types.is_categorical_dtype(X_cat[col]):
                    X_cat[col] = X_cat[col].astype(str)
                # Fill NaN in categorical features with 'MISSING'
                if X_cat[col].isna().any():
                    X_cat[col] = X_cat[col].fillna('MISSING')
        
        # Handle NaN in non-categorical columns (only truly numeric columns now)
        for col in X_cat.columns:
            if col not in actual_cat_features and X_cat[col].isna().any():
                # These should all be numeric at this point
                median_val = X_cat[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X_cat[col] = X_cat[col].fillna(median_val)
        
        # Get categorical feature names for CatBoost (use auto-detected list)
        cat_feature_names = [f for f in actual_cat_features if f in X_cat.columns]
        
        # Create and train model
        if is_categorical:
            model = cb.CatBoostClassifier(
                **self.model_params,
                cat_features=cat_feature_names
            )
        else:
            model = cb.CatBoostRegressor(
                **self.model_params,
                cat_features=cat_feature_names
            )
        
        model.fit(X_cat, y, verbose=False)
        
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
                # Convert any categorical dtype to regular dtype first
                for col in X_pred.columns:
                    if pd.api.types.is_categorical_dtype(X_pred[col]):
                        X_pred[col] = X_pred[col].astype(str)
                
                # Handle NaN in predictor features for XGBoost
                for col in X_pred.columns:
                    if X_pred[col].isna().any():
                        if col in cat_predictors:
                            # For categorical, use mode (most frequent value)
                            mode_val = X_pred[col].mode()[0] if not X_pred[col].mode().empty else 0
                            X_pred[col] = X_pred[col].fillna(mode_val)
                        else:
                            # For numerical, use median - but only if the column is actually numeric
                            if pd.api.types.is_numeric_dtype(X_pred[col]):
                                median_val = X_pred[col].median()
                                if pd.isna(median_val):
                                    median_val = 0.0
                                X_pred[col] = X_pred[col].fillna(median_val)
                            else:
                                # For non-numeric, use mode or a default value
                                mode_val = X_pred[col].mode()
                                if len(mode_val) > 0:
                                    X_pred[col] = X_pred[col].fillna(mode_val[0])
                                else:
                                    X_pred[col] = X_pred[col].fillna(0)
                
                # Convert ALL non-numeric columns to numeric codes (same as training)
                for col in X_pred.columns:
                    if not pd.api.types.is_numeric_dtype(X_pred[col]):
                        X_pred[col] = pd.Categorical(X_pred[col]).codes
                        X_pred[col] = X_pred[col].replace(-1, 0)
            
            elif self.model_type == "catboost":
                # First, convert any categorical dtype to string for CatBoost compatibility
                for col in X_pred.columns:
                    if pd.api.types.is_categorical_dtype(X_pred[col]):
                        X_pred[col] = X_pred[col].astype(str)
                
                # CRITICAL: Auto-detect non-numeric columns (same as training)
                # These need to be treated as categorical
                actual_cat_predictors = list(cat_predictors)
                for col in X_pred.columns:
                    if col not in actual_cat_predictors:
                        if not pd.api.types.is_numeric_dtype(X_pred[col]):
                            actual_cat_predictors.append(col)
                
                # Handle NaN in categorical features (declared + auto-detected)
                for col in actual_cat_predictors:
                    if col in X_pred.columns and X_pred[col].isna().any():
                        X_pred[col] = X_pred[col].fillna('MISSING')
                
                # Handle NaN in numerical features for CatBoost (only truly numeric now)
                for col in X_pred.columns:
                    if col not in actual_cat_predictors and X_pred[col].isna().any():
                        # These should all be numeric at this point
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
                # CatBoost may return probabilities, need to get class predictions
                if predicted.ndim > 1:
                    # If probabilities returned, get argmax for class prediction
                    predicted = predicted.argmax(axis=1)
                
                # Convert actual categorical values to numeric using stored mapping
                if not pd.api.types.is_numeric_dtype(actual):
                    # Use stored category mapping for consistent encoding
                    if feature_name in self.category_mappings:
                        # Create mapping from category to code
                        cat_map = {cat: idx for idx, cat in enumerate(self.category_mappings[feature_name])}
                        actual_str = actual.astype(str)
                        actual = np.array([cat_map.get(val, 99999) for val in actual_str])
                    else:
                        # Fallback to pd.Categorical if mapping not available
                        actual_codes = pd.Categorical(actual).codes
                        actual = np.where(actual_codes == -1, 99999, actual_codes)
                else:
                    # Already numeric, just ensure it's the right type
                    actual = np.array(actual, dtype=np.int32)
                
                # Handle remaining NaN/invalid in actual values - treat as always incorrect
                error = np.where(pd.isna(actual) | (actual == 99999), 1.0, (actual != predicted).astype(float))
            else:
                # Regression error (squared difference)
                # First, ensure actual is numeric (convert categorical to codes if needed)
                if not pd.api.types.is_numeric_dtype(actual):
                    actual_codes = pd.Categorical(actual).codes
                    actual = np.where(actual_codes == -1, 0, actual_codes)
                
                # Handle NaN in actual values - use 0 error (no information)
                actual_numeric = np.array(actual, dtype=np.float64)
                predicted_numeric = np.array(predicted, dtype=np.float64)
                error = np.where(pd.isna(actual_numeric), 0.0, (actual_numeric - predicted_numeric) ** 2)
            
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
            'model_params': self.model_params,
            'category_mappings': self.category_mappings
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
        instance.category_mappings = config_dict.get('category_mappings', {})
        
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
