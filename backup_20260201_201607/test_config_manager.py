"""
Tests for Configuration Manager
"""

import pytest
import yaml
from pathlib import Path

from src.config_manager import ConfigManager, create_default_config


class TestConfigManager:
    """Test suite for ConfigManager"""
    
    def test_create_default_config(self):
        """Test creating default configuration"""
        config = create_default_config()
        
        assert config is not None
        assert config.model.encoding_dim > 0
        assert len(config.model.hidden_layers) > 0
        assert config.training.batch_size > 0
        assert config.training.learning_rate > 0
    
    def test_load_config(self, tmp_path):
        """Test loading configuration from file"""
        # Create test config
        config_dict = {
            'data': {
                'train_path': 'data/train.parquet',
                'numerical_features': ['feature1', 'feature2'],
                'categorical_features': ['cat1'],
                'handle_missing': 'median'
            },
            'model': {
                'encoding_dim': 32,
                'hidden_layers': [128, 64],
                'dropout_rate': 0.3
            },
            'training': {
                'batch_size': 256,
                'learning_rate': 0.001,
                'max_epochs': 100
            },
            'evaluation': {
                'metrics': ['reconstruction_error']
            }
        }
        
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Load config
        manager = ConfigManager(config_path)
        config = manager.get_config()
        
        assert config.model.encoding_dim == 32
        assert config.model.hidden_layers == [128, 64]
        assert config.training.batch_size == 256
    
    def test_save_config(self, config_manager, tmp_path):
        """Test saving configuration"""
        output_path = tmp_path / "saved_config.yaml"
        config_manager.save_config(output_path)
        
        assert output_path.exists()
        
        # Load and verify
        with open(output_path, 'r') as f:
            loaded = yaml.safe_load(f)
        
        assert 'model' in loaded
        assert 'training' in loaded
    
    def test_update_config(self, config_manager):
        """Test updating configuration"""
        updates = {
            'model': {'encoding_dim': 64},
            'training': {'batch_size': 512}
        }
        
        config_manager.update_config(updates)
        config = config_manager.get_config()
        
        assert config.model.encoding_dim == 64
        assert config.training.batch_size == 512
    
    def test_get_all_features(self, config_manager):
        """Test getting all features"""
        features = config_manager.get_all_features()
        
        assert isinstance(features, list)
        assert len(features) > 0
    
    def test_get_feature_count(self, config_manager):
        """Test getting feature count"""
        count = config_manager.get_feature_count()
        
        assert isinstance(count, int)
        assert count > 0
    
    def test_validate_paths(self, config_manager):
        """Test path validation"""
        # Should create directories
        config_manager.validate_paths()
        
        paths_config = config_manager.config.paths
        
        assert Path(paths_config.models_dir).exists()
        assert Path(paths_config.outputs_dir).exists()
    
    def test_invalid_config_path(self):
        """Test loading from invalid path"""
        with pytest.raises(FileNotFoundError):
            ConfigManager("nonexistent/config.yaml")


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_invalid_handle_missing(self):
        """Test invalid missing value strategy"""
        with pytest.raises(ValueError):
            config_dict = {
                'data': {
                    'train_path': 'data.parquet',
                    'numerical_features': ['f1'],
                    'categorical_features': [],
                    'handle_missing': 'invalid_strategy'
                },
                'model': {
                    'encoding_dim': 32,
                    'hidden_layers': [64],
                    'dropout_rate': 0.3
                },
                'training': {
                    'batch_size': 256,
                    'learning_rate': 0.001,
                    'max_epochs': 100
                },
                'evaluation': {
                    'metrics': ['reconstruction_error']
                }
            }
            from src.config_manager import Config
            Config(**config_dict)
    
    def test_invalid_optimizer(self):
        """Test invalid optimizer"""
        with pytest.raises(ValueError):
            config_dict = {
                'data': {
                    'train_path': 'data.parquet',
                    'numerical_features': ['f1'],
                    'categorical_features': [],
                    'handle_missing': 'median'
                },
                'model': {
                    'encoding_dim': 32,
                    'hidden_layers': [64],
                    'dropout_rate': 0.3
                },
                'training': {
                    'batch_size': 256,
                    'learning_rate': 0.001,
                    'max_epochs': 100,
                    'optimizer': 'invalid_optimizer'
                },
                'evaluation': {
                    'metrics': ['reconstruction_error']
                }
            }
            from src.config_manager import Config
            Config(**config_dict)
    
    def test_negative_values(self):
        """Test negative value validation"""
        with pytest.raises(ValueError):
            config_dict = {
                'data': {
                    'train_path': 'data.parquet',
                    'numerical_features': ['f1'],
                    'categorical_features': [],
                    'handle_missing': 'median'
                },
                'model': {
                    'encoding_dim': -32,  # Invalid
                    'hidden_layers': [64],
                    'dropout_rate': 0.3
                },
                'training': {
                    'batch_size': 256,
                    'learning_rate': 0.001,
                    'max_epochs': 100
                },
                'evaluation': {
                    'metrics': ['reconstruction_error']
                }
            }
            from src.config_manager import Config
            Config(**config_dict)
