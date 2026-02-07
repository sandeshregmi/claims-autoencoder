"""
Tests for Preprocessing Module
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing import ClaimsPreprocessor, DataLoader


class TestClaimsPreprocessor:
    """Test suite for ClaimsPreprocessor"""
    
    def test_preprocessor_initialization(self, sample_config):
        """Test preprocessor initialization"""
        preprocessor = ClaimsPreprocessor(sample_config)
        
        assert preprocessor is not None
        assert not preprocessor.is_fitted
    
    def test_fit_transform(self, sample_config, sample_data):
        """Test fit and transform"""
        preprocessor = ClaimsPreprocessor(sample_config)
        X = preprocessor.fit_transform(sample_data)
        
        assert preprocessor.is_fitted
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] > 0
    
    def test_transform_without_fit(self, sample_config, sample_data):
        """Test that transform fails without fit"""
        preprocessor = ClaimsPreprocessor(sample_config)
        
        with pytest.raises(ValueError):
            preprocessor.transform(sample_data)
    
    def test_handle_missing_values(self, sample_config, sample_data):
        """Test missing value handling"""
        # Add missing values
        sample_data_copy = sample_data.copy()
        sample_data_copy.loc[0:10, 'claim_amount'] = np.nan
        
        preprocessor = ClaimsPreprocessor(sample_config)
        X = preprocessor.fit_transform(sample_data_copy)
        
        # Check no NaN values in output
        assert not np.any(np.isnan(X))
    
    def test_categorical_encoding(self, sample_config, sample_data):
        """Test categorical feature encoding"""
        preprocessor = ClaimsPreprocessor(sample_config)
        X = preprocessor.fit_transform(sample_data)
        
        # Check that label encoders were fitted
        assert len(preprocessor.label_encoders) > 0
        
        for col in sample_config.data.categorical_features:
            assert col in preprocessor.label_encoders
    
    def test_get_feature_names(self, sample_config, sample_data):
        """Test getting feature names"""
        preprocessor = ClaimsPreprocessor(sample_config)
        preprocessor.fit(sample_data)
        
        feature_names = preprocessor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
    
    def test_get_feature_count(self, sample_config, sample_data):
        """Test getting feature count"""
        preprocessor = ClaimsPreprocessor(sample_config)
        preprocessor.fit(sample_data)
        
        count = preprocessor.get_feature_count()
        
        assert isinstance(count, int)
        assert count > 0
    
    def test_inverse_transform(self, sample_config, sample_data):
        """Test inverse transform"""
        preprocessor = ClaimsPreprocessor(sample_config)
        X = preprocessor.fit_transform(sample_data)
        
        X_inv = preprocessor.inverse_transform(X)
        
        assert X_inv.shape == X.shape
    
    def test_save_load(self, sample_config, sample_data, tmp_path):
        """Test saving and loading preprocessor"""
        # Fit preprocessor
        preprocessor = ClaimsPreprocessor(sample_config)
        X_original = preprocessor.fit_transform(sample_data)
        
        # Save
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        assert save_path.exists()
        
        # Load
        loaded_preprocessor = ClaimsPreprocessor.load(str(save_path), sample_config)
        
        assert loaded_preprocessor.is_fitted
        
        # Transform with loaded preprocessor
        X_loaded = loaded_preprocessor.transform(sample_data)
        
        # Should produce same output
        np.testing.assert_array_almost_equal(X_original, X_loaded)
    
    def test_handle_new_categories(self, sample_config, sample_data):
        """Test handling of unseen categories"""
        # Fit on subset
        train_data = sample_data[:500].copy()
        preprocessor = ClaimsPreprocessor(sample_config)
        preprocessor.fit(train_data)
        
        # Add new category
        test_data = sample_data[500:].copy()
        test_data.loc[test_data.index[0], 'claim_type'] = 'new_category'
        
        # Should not raise error
        X = preprocessor.transform(test_data)
        assert X is not None
    
    def test_scaling(self, sample_config, sample_data):
        """Test that features are scaled"""
        preprocessor = ClaimsPreprocessor(sample_config)
        X = preprocessor.fit_transform(sample_data)
        
        # Check that features are roughly standardized
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        
        # Most features should have mean close to 0 and std close to 1
        # (using RobustScaler, so not exactly standard normal)
        assert np.median(np.abs(means)) < 1.0
    
    def test_handle_outliers(self, sample_config, sample_data):
        """Test outlier handling"""
        preprocessor = ClaimsPreprocessor(sample_config)
        
        # Process with outlier clipping
        processed_data = preprocessor.handle_outliers(
            sample_data,
            method='clip',
            threshold=3.0
        )
        
        assert len(processed_data) == len(sample_data)


class TestDataLoader:
    """Test suite for DataLoader"""
    
    def test_dataloader_initialization(self):
        """Test data loader initialization"""
        X = np.random.randn(100, 20)
        loader = DataLoader(X, batch_size=16, shuffle=True)
        
        assert loader is not None
        assert loader.batch_size == 16
        assert loader.n_samples == 100
    
    def test_dataloader_iteration(self):
        """Test iterating through data loader"""
        X = np.random.randn(100, 20)
        loader = DataLoader(X, batch_size=16, shuffle=False)
        
        batches = list(loader)
        
        # Should have ceil(100/16) = 7 batches
        assert len(batches) == 7
        
        # Check batch shapes
        for i, batch in enumerate(batches[:-1]):
            assert batch.shape == (16, 20)
        
        # Last batch might be smaller
        assert batches[-1].shape[0] <= 16
    
    def test_dataloader_shuffle(self):
        """Test data shuffling"""
        X = np.arange(100).reshape(100, 1)
        loader = DataLoader(X, batch_size=10, shuffle=True)
        
        # Get first batch twice
        batches1 = list(loader)
        batches2 = list(loader)
        
        # Should be different due to shuffling
        assert not np.array_equal(batches1[0], batches2[0])
    
    def test_dataloader_len(self):
        """Test data loader length"""
        X = np.random.randn(100, 20)
        loader = DataLoader(X, batch_size=16)
        
        assert len(loader) == 7


class TestFeatureEngineering:
    """Test feature engineering functionality"""
    
    def test_feature_interactions(self, sample_config, sample_data):
        """Test feature interaction creation"""
        # Enable feature interactions
        sample_config.data.feature_interactions = {
            'enabled': True,
            'pairs': [
                ['claim_amount', 'patient_age'],
                ['num_previous_claims', 'average_claim_amount']
            ]
        }
        
        preprocessor = ClaimsPreprocessor(sample_config)
        X = preprocessor.fit_transform(sample_data)
        
        feature_names = preprocessor.get_feature_names()
        
        # Check interaction features were created
        assert any('_x_' in name for name in feature_names)
    
    def test_feature_statistics(self, sample_config, sample_data):
        """Test feature statistics computation"""
        preprocessor = ClaimsPreprocessor(sample_config)
        preprocessor.fit(sample_data)
        
        stats = preprocessor.feature_stats
        
        assert len(stats) > 0
        
        # Check statistics structure
        for feature, feature_stats in stats.items():
            assert 'mean' in feature_stats
            assert 'std' in feature_stats
            assert 'min' in feature_stats
            assert 'max' in feature_stats
