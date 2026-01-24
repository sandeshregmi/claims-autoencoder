"""
Tests for Claims Tree-Based Fraud Detection Models

Author: ML Engineering Team
Date: 2026-01-22
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tree_models import ClaimsTreeAutoencoder, create_ensemble_fraud_scores


@pytest.fixture
def sample_claims():
    """Create sample claims data for testing."""
    np.random.seed(42)
    
    data = {
        'claim_amount': np.random.uniform(1000, 10000, 100),
        'patient_age': np.random.randint(25, 70, 100),
        'provider_experience_years': np.random.randint(5, 30, 100),
        'claim_type': np.random.choice(['routine', 'emergency'], 100),
        'provider_specialty': np.random.choice(['general', 'specialist'], 100),
        'diagnosis_code': np.random.choice(['D001', 'D002'], 100),
    }
    
    df = pd.DataFrame(data)
    
    categorical_features = ['claim_type', 'provider_specialty', 'diagnosis_code']
    numerical_features = ['claim_amount', 'patient_age', 'provider_experience_years']
    
    return df, categorical_features, numerical_features


class TestClaimsTreeAutoencoder:
    """Test suite for ClaimsTreeAutoencoder."""
    
    def test_initialization_xgboost(self):
        """Test XGBoost initialization."""
        model = ClaimsTreeAutoencoder(model_type='xgboost')
        
        assert model.model_type == 'xgboost'
        assert model.models == {}
        assert model.feature_names == []
    
    def test_initialization_catboost(self):
        """Test CatBoost initialization."""
        model = ClaimsTreeAutoencoder(model_type='catboost')
        
        assert model.model_type == 'catboost'
        assert model.models == {}
        assert model.feature_names == []
    
    def test_initialization_invalid_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError, match="model_type must be"):
            ClaimsTreeAutoencoder(model_type='invalid_model')
    
    def test_fit_xgboost(self, sample_claims):
        """Test XGBoost training."""
        X, cat_features, num_features = sample_claims
        
        model = ClaimsTreeAutoencoder(model_type='xgboost', n_estimators=10)
        model.fit(X, categorical_features=cat_features,
                 numerical_features=num_features, verbose=False)
        
        assert len(model.models) == len(X.columns)
        assert model.feature_names == list(X.columns)
        assert model.categorical_features == cat_features
        assert model.numerical_features == num_features
    
    def test_fit_catboost(self, sample_claims):
        """Test CatBoost training."""
        X, cat_features, num_features = sample_claims
        
        model = ClaimsTreeAutoencoder(model_type='catboost', iterations=10)
        model.fit(X, categorical_features=cat_features,
                 numerical_features=num_features, verbose=False)
        
        assert len(model.models) == len(X.columns)
        assert model.feature_names == list(X.columns)
    
    def test_predict(self, sample_claims):
        """Test prediction generation."""
        X, cat_features, num_features = sample_claims
        
        model = ClaimsTreeAutoencoder(model_type='xgboost', n_estimators=10)
        model.fit(X, categorical_features=cat_features,
                 numerical_features=num_features, verbose=False)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X.columns)
        for feature_name, pred in predictions.items():
            assert len(pred) == len(X)
    
    def test_predict_not_trained(self, sample_claims):
        """Test prediction before training raises error."""
        X, _, _ = sample_claims
        
        model = ClaimsTreeAutoencoder(model_type='xgboost')
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(X)
    
    def test_compute_fraud_scores_l2(self, sample_claims):
        """Test fraud score computation with L2 norm."""
        X, cat_features, num_features = sample_claims
        
        model = ClaimsTreeAutoencoder(model_type='xgboost', n_estimators=10)
        model.fit(X, categorical_features=cat_features,
                 numerical_features=num_features, verbose=False)
        
        scores, per_feature_errors = model.compute_fraud_scores(X, method='l2')
        
        assert len(scores) == len(X)
        assert len(per_feature_errors) == len(X.columns)
        assert all(scores >= 0)
    
    def test_compute_fraud_scores_l1(self, sample_claims):
        """Test fraud score computation with L1 norm."""
        X, cat_features, num_features = sample_claims
        
        model = ClaimsTreeAutoencoder(model_type='xgboost', n_estimators=10)
        model.fit(X, categorical_features=cat_features,
                 numerical_features=num_features, verbose=False)
        
        scores, _ = model.compute_fraud_scores(X, method='l1')
        
        assert len(scores) == len(X)
        assert all(scores >= 0)
    
    def test_compute_fraud_scores_max(self, sample_claims):
        """Test fraud score computation with max method."""
        X, cat_features, num_features = sample_claims
        
        model = ClaimsTreeAutoencoder(model_type='xgboost', n_estimators=10)
        model.fit(X, categorical_features=cat_features,
                 numerical_features=num_features, verbose=False)
        
        scores, _ = model.compute_fraud_scores(X, method='max')
        
        assert len(scores) == len(X)
        assert all(scores >= 0)
    
    def test_get_feature_importance(self, sample_claims):
        """Test feature importance extraction."""
        X, cat_features, num_features = sample_claims
        
        model = ClaimsTreeAutoencoder(model_type='xgboost', n_estimators=10)
        model.fit(X, categorical_features=cat_features,
                 numerical_features=num_features, verbose=False)
        
        importances = model.get_feature_importance()
        
        assert len(importances) == len(X.columns)
        for target_feature, feature_importances in importances.items():
            assert len(feature_importances) == len(X.columns) - 1
    
    def test_get_top_fraud_indicators(self, sample_claims):
        """Test getting top fraud indicators."""
        X, cat_features, num_features = sample_claims
        
        model = ClaimsTreeAutoencoder(model_type='xgboost', n_estimators=10)
        model.fit(X, categorical_features=cat_features,
                 numerical_features=num_features, verbose=False)
        
        top_indicators = model.get_top_fraud_indicators('claim_amount', top_k=2)
        
        assert len(top_indicators) == 2
        assert all(isinstance(item, tuple) for item in top_indicators)
        
        # Check descending order
        importances = [imp for _, imp in top_indicators]
        assert importances == sorted(importances, reverse=True)
    
    def test_save_load(self, sample_claims):
        """Test model save and load."""
        X, cat_features, num_features = sample_claims
        
        # Train model
        model = ClaimsTreeAutoencoder(model_type='xgboost', n_estimators=10)
        model.fit(X, categorical_features=cat_features,
                 numerical_features=num_features, verbose=False)
        
        # Get original predictions
        original_scores, _ = model.compute_fraud_scores(X)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            model.save(str(save_path))
            
            # Load model
            loaded_model = ClaimsTreeAutoencoder.load(str(save_path))
            
            # Get loaded predictions
            loaded_scores, _ = loaded_model.compute_fraud_scores(X)
            
            # Verify consistency
            np.testing.assert_array_almost_equal(original_scores, loaded_scores, decimal=10)


class TestEnsemble:
    """Test ensemble fraud detection."""
    
    def test_ensemble_equal_weights(self, sample_claims):
        """Test ensemble with equal weights."""
        X, cat_features, num_features = sample_claims
        
        # Train models
        models = {}
        for model_type in ['xgboost', 'catboost']:
            model = ClaimsTreeAutoencoder(
                model_type=model_type,
                n_estimators=10 if model_type == 'xgboost' else None,
                iterations=10 if model_type == 'catboost' else None
            )
            model.fit(X, categorical_features=cat_features,
                     numerical_features=num_features, verbose=False)
            models[model_type] = model
        
        # Create ensemble
        ensemble_scores = create_ensemble_fraud_scores(models, X)
        
        assert len(ensemble_scores) == len(X)
        assert all(ensemble_scores >= 0)
    
    def test_ensemble_custom_weights(self, sample_claims):
        """Test ensemble with custom weights."""
        X, cat_features, num_features = sample_claims
        
        # Train models
        models = {}
        for model_type in ['xgboost', 'catboost']:
            model = ClaimsTreeAutoencoder(
                model_type=model_type,
                n_estimators=10 if model_type == 'xgboost' else None,
                iterations=10 if model_type == 'catboost' else None
            )
            model.fit(X, categorical_features=cat_features,
                     numerical_features=num_features, verbose=False)
            models[model_type] = model
        
        # Create ensemble with custom weights
        weights = {'xgboost': 0.7, 'catboost': 0.3}
        ensemble_scores = create_ensemble_fraud_scores(models, X, weights=weights)
        
        assert len(ensemble_scores) == len(X)


def test_end_to_end_fraud_detection(sample_claims):
    """Test complete fraud detection workflow."""
    X, cat_features, num_features = sample_claims
    
    # 1. Train model
    model = ClaimsTreeAutoencoder(model_type='xgboost', n_estimators=10)
    model.fit(X, categorical_features=cat_features,
             numerical_features=num_features, verbose=False)
    
    # 2. Make predictions
    predictions = model.predict(X)
    assert len(predictions) == len(X.columns)
    
    # 3. Compute fraud scores
    scores, per_feature_errors = model.compute_fraud_scores(X)
    assert len(scores) == len(X)
    
    # 4. Analyze indicators
    importances = model.get_feature_importance()
    assert len(importances) > 0
    
    # 5. Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model"
        model.save(str(save_path))
        loaded_model = ClaimsTreeAutoencoder.load(str(save_path))
        
        # Verify loaded model works
        loaded_scores, _ = loaded_model.compute_fraud_scores(X)
        np.testing.assert_array_almost_equal(scores, loaded_scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
