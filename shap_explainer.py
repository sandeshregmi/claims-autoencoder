"""
SHAP-based Local Explainability for Tree Models

Provides local feature importance explanations for individual claim predictions
using SHAP (SHapley Additive exPlanations).

Author: ML Engineering Team  
Date: 2026-01-24
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Tuple, Any

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logger = logging.getLogger(__name__)


class ClaimsShapExplainer:
    """
    SHAP-based explainer for tree-based fraud detection models.
    
    Provides local (per-claim) and global feature importance using SHAP values.
    Works with both XGBoost and CatBoost models.
    """
    
    def __init__(self, model, feature_names: List[str], categorical_features: List[str]):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained ClaimsTreeAutoencoder
            feature_names: List of feature names
            categorical_features: List of categorical feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP not installed. Install with: pip install shap"
            )
        
        self.model = model
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.numerical_features = [f for f in feature_names if f not in categorical_features]
        
        # Store explainers for each target feature
        self.explainers: Dict[str, Any] = {}
        
        logger.info(f"Initialized SHAP explainer for {len(feature_names)} features")
    
    def create_explainers(self, X_background: pd.DataFrame, max_samples: int = 100):
        """
        Create SHAP explainers for all models.
        
        Args:
            X_background: Background data for SHAP (typically training set)
            max_samples: Maximum samples to use for background (for speed)
        """
        logger.info("Creating SHAP explainers for all features...")
        
        # Sample background data if too large
        if len(X_background) > max_samples:
            X_background = X_background.sample(n=max_samples, random_state=42)
        
        for target_feature in self.feature_names:
            # Get predictor features
            predictor_features = [f for f in self.feature_names if f != target_feature]
            X_bg = X_background[predictor_features].copy()
            
            # Preprocess same as training/prediction
            X_bg = self._preprocess_for_model(X_bg, predictor_features)
            
            # Get model for this target
            model = self.model.models[target_feature]
            
            # Create TreeExplainer (fast for tree models)
            if self.model.model_type == "xgboost":
                explainer = shap.TreeExplainer(model, X_bg)
            elif self.model.model_type == "catboost":
                explainer = shap.TreeExplainer(model)
            
            self.explainers[target_feature] = {
                'explainer': explainer,
                'predictor_features': predictor_features
            }
        
        logger.info(f"✓ Created {len(self.explainers)} SHAP explainers")
    
    def _preprocess_for_model(self, X: pd.DataFrame, predictor_features: List[str]) -> pd.DataFrame:
        """Preprocess data same as model training/prediction."""
        X_processed = X.copy()
        
        cat_predictors = [f for f in predictor_features if f in self.categorical_features]
        
        if self.model.model_type == "xgboost":
            # Convert categorical to codes
            for col in cat_predictors:
                if col in X_processed.columns:
                    codes = pd.Categorical(X_processed[col]).codes
                    X_processed[col] = np.where(codes == -1, 0, codes)
            
            # Impute NaN
            for col in X_processed.columns:
                if X_processed[col].isna().any():
                    if col in cat_predictors:
                        mode_val = X_processed[col].mode()[0] if not X_processed[col].mode().empty else 0
                        X_processed[col] = X_processed[col].fillna(mode_val)
                    else:
                        median_val = X_processed[col].median()
                        X_processed[col] = X_processed[col].fillna(median_val if not pd.isna(median_val) else 0.0)
        
        elif self.model.model_type == "catboost":
            # Replace NaN in categorical with 'MISSING'
            for col in cat_predictors:
                if col in X_processed.columns and X_processed[col].isna().any():
                    X_processed[col] = X_processed[col].fillna('MISSING')
            
            # Impute numerical NaN
            for col in X_processed.columns:
                if col not in cat_predictors and X_processed[col].isna().any():
                    median_val = X_processed[col].median()
                    X_processed[col] = X_processed[col].fillna(median_val if not pd.isna(median_val) else 0.0)
        
        return X_processed
    
    def explain_claim(
        self, 
        claim_data: pd.DataFrame, 
        target_feature: str,
        plot: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Explain prediction for a single claim.
        
        Args:
            claim_data: Single claim data (1 row DataFrame)
            target_feature: Which feature prediction to explain
            plot: Whether to show waterfall plot
            
        Returns:
            Tuple of (shap_values, feature_contributions DataFrame)
        """
        if target_feature not in self.explainers:
            raise ValueError(f"No explainer for feature: {target_feature}")
        
        explainer_info = self.explainers[target_feature]
        explainer = explainer_info['explainer']
        predictor_features = explainer_info['predictor_features']
        
        # Prepare data
        X_claim = claim_data[predictor_features].copy()
        X_claim = self._preprocess_for_model(X_claim, predictor_features)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_claim)
        
        # Handle multi-class case (CatBoost may return array of arrays)
        if isinstance(shap_values, list):
            # For multi-class, use values for predicted class
            model = self.model.models[target_feature]
            prediction = model.predict(X_claim)
            if hasattr(prediction, 'ndim') and prediction.ndim > 1:
                predicted_class = prediction.argmax(axis=1)[0]
            else:
                predicted_class = int(prediction[0])
            shap_values = shap_values[predicted_class]
        
        # Create feature contribution DataFrame
        if shap_values.ndim > 1:
            shap_values = shap_values[0]
        
        contributions = pd.DataFrame({
            'feature': predictor_features,
            'value': X_claim.iloc[0].values,
            'shap_value': shap_values,
            'abs_shap': np.abs(shap_values)
        }).sort_values('abs_shap', ascending=False)
        
        # Plot if requested
        if plot:
            self._plot_waterfall(
                shap_values, 
                X_claim.iloc[0], 
                predictor_features,
                target_feature,
                explainer.expected_value
            )
        
        return shap_values, contributions
    
    def explain_top_frauds(
        self,
        X: pd.DataFrame,
        fraud_scores: np.ndarray,
        top_k: int = 10,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Explain top-k most fraudulent claims.
        
        Args:
            X: All claims data
            fraud_scores: Fraud scores for all claims
            top_k: Number of top fraudulent claims to explain
            save_path: Optional path to save explanations
            
        Returns:
            DataFrame with explanations for top fraudulent claims
        """
        # Get top-k fraudulent claims
        top_indices = np.argsort(fraud_scores)[-top_k:][::-1]
        
        explanations = []
        
        for rank, idx in enumerate(top_indices, 1):
            claim = X.iloc[[idx]]
            fraud_score = fraud_scores[idx]
            
            # Get SHAP values for most important features
            claim_explanations = {
                'rank': rank,
                'claim_index': idx,
                'fraud_score': fraud_score
            }
            
            # Add actual feature values
            for col in self.feature_names:
                claim_explanations[f'actual_{col}'] = claim[col].values[0]
            
            explanations.append(claim_explanations)
        
        explanations_df = pd.DataFrame(explanations)
        
        if save_path:
            explanations_df.to_csv(save_path, index=False)
            logger.info(f"Saved explanations to {save_path}")
        
        return explanations_df
    
    def get_global_feature_importance(
        self,
        X: pd.DataFrame,
        max_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Get global feature importance across all predictions.
        
        Args:
            X: Data to compute importance on
            max_samples: Maximum samples to use
            
        Returns:
            DataFrame with feature importance rankings
        """
        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)
        
        all_importances = []
        
        for target_feature in self.feature_names:
            explainer_info = self.explainers[target_feature]
            predictor_features = explainer_info['predictor_features']
            explainer = explainer_info['explainer']
            
            # Prepare data
            X_pred = X[predictor_features].copy()
            X_pred = self._preprocess_for_model(X_pred, predictor_features)
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_pred)
            
            # Handle multi-class
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first class
            
            # Average absolute SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            for i, feature in enumerate(predictor_features):
                all_importances.append({
                    'target_feature': target_feature,
                    'predictor_feature': feature,
                    'mean_abs_shap': mean_abs_shap[i]
                })
        
        importance_df = pd.DataFrame(all_importances)
        
        # Aggregate across all targets
        global_importance = importance_df.groupby('predictor_feature')['mean_abs_shap'].mean()
        global_importance = global_importance.sort_values(ascending=False)
        
        return pd.DataFrame({
            'feature': global_importance.index,
            'importance': global_importance.values
        })
    
    def _plot_waterfall(
        self,
        shap_values: np.ndarray,
        feature_values: pd.Series,
        feature_names: List[str],
        target_feature: str,
        expected_value: float
    ):
        """Plot SHAP waterfall for single prediction."""
        # Sort by absolute SHAP value
        indices = np.argsort(np.abs(shap_values))[::-1][:10]  # Top 10
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Cumulative sum for waterfall
        cumsum = expected_value
        positions = []
        colors = []
        
        for i, idx in enumerate(indices):
            positions.append(cumsum)
            cumsum += shap_values[idx]
            colors.append('red' if shap_values[idx] > 0 else 'blue')
        
        # Plot bars
        feature_labels = [feature_names[i] for i in indices]
        shap_vals = [shap_values[i] for i in indices]
        
        ax.barh(range(len(indices)), shap_vals, color=colors, alpha=0.7)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(feature_labels)
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title(f'Feature Impact on Predicting: {target_feature}')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_summary(
        self,
        X: pd.DataFrame,
        target_feature: str,
        max_samples: int = 1000,
        plot_type: str = 'bar'
    ):
        """
        Plot SHAP summary for a target feature.
        
        Args:
            X: Data to compute SHAP on
            target_feature: Which feature to explain
            max_samples: Maximum samples
            plot_type: 'bar' or 'dot'
        """
        if target_feature not in self.explainers:
            raise ValueError(f"No explainer for feature: {target_feature}")
        
        explainer_info = self.explainers[target_feature]
        predictor_features = explainer_info['predictor_features']
        explainer = explainer_info['explainer']
        
        # Sample data
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
        else:
            X_sample = X.copy()
        
        # Prepare data
        X_pred = X_sample[predictor_features].copy()
        X_pred = self._preprocess_for_model(X_pred, predictor_features)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_pred)
        
        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(shap_values, X_pred, plot_type='bar', show=True)
        else:
            shap.summary_plot(shap_values, X_pred, show=True)
        
        plt.title(f'SHAP Summary for: {target_feature}')
        plt.tight_layout()
