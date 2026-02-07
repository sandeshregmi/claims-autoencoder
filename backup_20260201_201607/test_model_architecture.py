"""
Tests for Model Architecture
"""

import pytest
import torch
import numpy as np

from src.model_architecture import (
    ClaimsAutoencoder,
    create_model_from_config,
    EarlyStopping
)


class TestClaimsAutoencoder:
    """Test suite for ClaimsAutoencoder"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = ClaimsAutoencoder(
            input_dim=20,
            encoding_dim=32,
            hidden_layers=[128, 64],
            dropout_rate=0.3,
            batch_norm=True,
            activation='relu'
        )
        
        assert model is not None
        assert model.input_dim == 20
        assert model.encoding_dim == 32
        assert model.hidden_layers == [128, 64]
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = ClaimsAutoencoder(
            input_dim=20,
            encoding_dim=32,
            hidden_layers=[128, 64],
            dropout_rate=0.3
        )
        
        batch_size = 16
        x = torch.randn(batch_size, 20)
        
        reconstruction, encoding = model(x)
        
        assert reconstruction.shape == (batch_size, 20)
        assert encoding.shape == (batch_size, 32)
    
    def test_encode_decode(self):
        """Test encode and decode separately"""
        model = ClaimsAutoencoder(
            input_dim=20,
            encoding_dim=32,
            hidden_layers=[64]
        )
        
        x = torch.randn(10, 20)
        
        # Encode
        encoding = model.encode(x)
        assert encoding.shape == (10, 32)
        
        # Decode
        reconstruction = model.decode(encoding)
        assert reconstruction.shape == (10, 20)
    
    def test_reconstruction_error(self):
        """Test reconstruction error computation"""
        model = ClaimsAutoencoder(
            input_dim=20,
            encoding_dim=32,
            hidden_layers=[64]
        )
        
        x = torch.randn(10, 20)
        
        # Per-sample error
        error = model.compute_reconstruction_error(x, reduction='none')
        assert error.shape == (10,)
        
        # Mean error
        error = model.compute_reconstruction_error(x, reduction='mean')
        assert error.shape == torch.Size([])
    
    def test_anomaly_score(self):
        """Test anomaly score computation"""
        model = ClaimsAutoencoder(
            input_dim=20,
            encoding_dim=32,
            hidden_layers=[64]
        )
        
        x = torch.randn(10, 20)
        scores = model.get_anomaly_score(x)
        
        assert scores.shape == (10,)
        assert torch.all(scores >= 0)
    
    def test_count_parameters(self):
        """Test parameter counting"""
        model = ClaimsAutoencoder(
            input_dim=20,
            encoding_dim=32,
            hidden_layers=[128, 64]
        )
        
        param_count = model.count_parameters()
        assert param_count > 0
    
    def test_freeze_unfreeze(self):
        """Test freezing and unfreezing layers"""
        model = ClaimsAutoencoder(
            input_dim=20,
            encoding_dim=32,
            hidden_layers=[64]
        )
        
        # Freeze encoder
        model.freeze_encoder()
        for param in model.encoder.parameters():
            assert not param.requires_grad
        
        # Unfreeze encoder
        model.unfreeze_encoder()
        for param in model.encoder.parameters():
            assert param.requires_grad
    
    def test_different_activations(self):
        """Test different activation functions"""
        activations = ['relu', 'tanh', 'leaky_relu', 'elu']
        
        for activation in activations:
            model = ClaimsAutoencoder(
                input_dim=20,
                encoding_dim=32,
                hidden_layers=[64],
                activation=activation
            )
            
            x = torch.randn(5, 20)
            reconstruction, encoding = model(x)
            
            assert reconstruction.shape == (5, 20)
    
    def test_create_model_from_config(self, sample_config):
        """Test creating model from config"""
        model = create_model_from_config(sample_config, input_dim=20)
        
        assert model is not None
        assert model.input_dim == 20
        assert model.encoding_dim == sample_config.model.encoding_dim


class TestEarlyStopping:
    """Test suite for EarlyStopping"""
    
    def test_early_stopping_initialization(self):
        """Test early stopping initialization"""
        es = EarlyStopping(patience=5, min_delta=0.001, mode='min')
        
        assert es.patience == 5
        assert es.min_delta == 0.001
        assert es.counter == 0
        assert not es.early_stop
    
    def test_early_stopping_trigger(self):
        """Test early stopping trigger"""
        es = EarlyStopping(patience=3, mode='min')
        
        # Improving scores
        assert not es(1.0)
        assert not es(0.9)
        assert not es(0.8)
        
        # No improvement
        assert not es(0.85)
        assert not es(0.85)
        assert es(0.85)  # Should trigger
        
        assert es.early_stop
    
    def test_early_stopping_no_trigger(self):
        """Test that early stopping doesn't trigger with improvement"""
        es = EarlyStopping(patience=3, mode='min')
        
        for score in [1.0, 0.9, 0.8, 0.7, 0.6]:
            assert not es(score)
        
        assert not es.early_stop
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in maximize mode"""
        es = EarlyStopping(patience=2, mode='max')
        
        # Improving
        assert not es(0.5)
        assert not es(0.6)
        assert not es(0.7)
        
        # No improvement
        assert not es(0.65)
        assert es(0.65)  # Should trigger
        
        assert es.early_stop


class TestModelTraining:
    """Test model training functionality"""
    
    def test_model_gradient_flow(self):
        """Test that gradients flow correctly"""
        model = ClaimsAutoencoder(
            input_dim=20,
            encoding_dim=32,
            hidden_layers=[64]
        )
        
        x = torch.randn(10, 20, requires_grad=True)
        reconstruction, _ = model(x)
        
        loss = torch.mean((reconstruction - x) ** 2)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_eval_mode(self):
        """Test model evaluation mode"""
        model = ClaimsAutoencoder(
            input_dim=20,
            encoding_dim=32,
            hidden_layers=[64]
        )
        
        x = torch.randn(10, 20)
        
        # Training mode
        model.train()
        out1, _ = model(x)
        
        # Eval mode
        model.eval()
        with torch.no_grad():
            out2, _ = model(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(out1, out2)
