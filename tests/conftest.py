"""
Pytest configuration and fixtures
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.config_manager import ConfigManager, create_default_config
from src.model_architecture import ClaimsAutoencoder
from src.preprocessing import ClaimsPreprocessor


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return create_default_config()


@pytest.fixture
def config_manager(tmp_path):
    """Create a config manager with temporary config file."""
    config = create_default_config()
    config_path = tmp_path / "test_config.yaml"
    
    manager = ConfigManager()
    manager.config = config
    manager.save_config(config_path)
    
    return ConfigManager(config_path)


@pytest.fixture
def sample_data():
    """Generate sample claims data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'claim_amount': np.random.gamma(2, 1000, n_samples),
        'patient_age': np.random.randint(18, 90, n_samples),
        'provider_experience_years': np.random.randint(0, 40, n_samples),
        'days_since_last_claim': np.random.exponential(30, n_samples),
        'num_previous_claims': np.random.poisson(2, n_samples),
        'average_claim_amount': np.random.gamma(2, 800, n_samples),
        'claim_duration_days': np.random.randint(1, 60, n_samples),
        'claim_type': np.random.choice(['medical', 'dental', 'vision'], n_samples),
        'provider_specialty': np.random.choice(['general', 'specialist'], n_samples),
        'diagnosis_code': np.random.choice(['D001', 'D002', 'D003'], n_samples),
        'procedure_code': np.random.choice(['P001', 'P002'], n_samples),
        'patient_gender': np.random.choice(['M', 'F'], n_samples),
        'geographic_region': np.random.choice(['North', 'South'], n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def preprocessed_data(sample_config, sample_data):
    """Generate preprocessed data for testing."""
    preprocessor = ClaimsPreprocessor(sample_config)
    X = preprocessor.fit_transform(sample_data)
    return X


@pytest.fixture
def sample_model(sample_config):
    """Create a sample model for testing."""
    model = ClaimsAutoencoder(
        input_dim=20,
        encoding_dim=sample_config.model.encoding_dim,
        hidden_layers=sample_config.model.hidden_layers,
        dropout_rate=sample_config.model.dropout_rate,
        batch_norm=sample_config.model.batch_norm,
        activation=sample_config.model.activation
    )
    return model


@pytest.fixture
def device():
    """Get device for testing."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test outputs."""
    test_dir = tmp_path / "test_outputs"
    test_dir.mkdir()
    return test_dir
