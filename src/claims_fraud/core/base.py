"""
Model Architecture Module for Claims Autoencoder
Defines the autoencoder neural network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging


logger = logging.getLogger(__name__)


class ClaimsAutoencoder(nn.Module):
    """
    Autoencoder architecture for claims anomaly detection.
    
    Architecture:
    - Encoder: Input -> Hidden Layers -> Bottleneck (encoding)
    - Decoder: Bottleneck -> Hidden Layers -> Output (reconstruction)
    
    Features:
    - Configurable hidden layers
    - Dropout for regularization
    - Batch normalization
    - Multiple activation functions
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        hidden_layers: List[int],
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of encoding (bottleneck)
            hidden_layers: List of hidden layer sizes for encoder
            dropout_rate: Dropout probability
            batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'tanh', 'leaky_relu')
        """
        super(ClaimsAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.activation_name = activation
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder (symmetric to encoder)
        self.decoder = self._build_decoder()
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized ClaimsAutoencoder: input_dim={input_dim}, "
                   f"encoding_dim={encoding_dim}, hidden_layers={hidden_layers}")
    
    def _build_encoder(self) -> nn.Sequential:
        """
        Build encoder network.
        
        Returns:
            Sequential encoder model
        """
        layers = []
        
        # Input to first hidden layer
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation())
            layers.append(nn.Dropout(self.dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final encoding layer
        layers.append(nn.Linear(prev_dim, self.encoding_dim))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Sequential:
        """
        Build decoder network (symmetric to encoder).
        
        Returns:
            Sequential decoder model
        """
        layers = []
        
        # Bottleneck to first decoder layer
        prev_dim = self.encoding_dim
        
        # Reverse the hidden layers for decoder
        decoder_hidden = list(reversed(self.hidden_layers))
        
        for hidden_dim in decoder_hidden:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation())
            layers.append(nn.Dropout(self.dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final reconstruction layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self) -> nn.Module:
        """
        Get activation function.
        
        Returns:
            Activation module
        """
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
        }
        
        return activations.get(self.activation_name, nn.ReLU())
    
    def _init_weights(self, module):
        """
        Initialize network weights.
        
        Args:
            module: Network module
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to bottleneck representation.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            Encoded representation [batch_size, encoding_dim]
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode bottleneck representation to reconstruction.
        
        Args:
            z: Encoded tensor [batch_size, encoding_dim]
        
        Returns:
            Reconstructed output [batch_size, input_dim]
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            Tuple of (reconstruction, encoding)
        """
        encoding = self.encode(x)
        reconstruction = self.decode(encoding)
        return reconstruction, encoding
    
    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Compute reconstruction error (MSE).
        
        Args:
            x: Input tensor [batch_size, input_dim]
            reduction: Reduction method ('none', 'mean', 'sum')
        
        Returns:
            Reconstruction error tensor
        """
        reconstruction, _ = self.forward(x)
        
        if reduction == 'none':
            # Per-sample error
            error = torch.mean((x - reconstruction) ** 2, dim=1)
        elif reduction == 'mean':
            error = torch.mean((x - reconstruction) ** 2)
        elif reduction == 'sum':
            error = torch.sum((x - reconstruction) ** 2)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        return error
    
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get anomaly scores (reconstruction errors) for input.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            Anomaly scores [batch_size]
        """
        self.eval()
        with torch.no_grad():
            scores = self.compute_reconstruction_error(x, reduction='none')
        return scores
    
    def count_parameters(self) -> int:
        """
        Count total trainable parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_sizes(self) -> dict:
        """
        Get sizes of all layers.
        
        Returns:
            Dictionary with layer information
        """
        return {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'encoder_hidden': self.hidden_layers,
            'decoder_hidden': list(reversed(self.hidden_layers)),
            'total_parameters': self.count_parameters(),
        }
    
    def freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")
    
    def freeze_decoder(self):
        """Freeze decoder parameters"""
        for param in self.decoder.parameters():
            param.requires_grad = False
        logger.info("Decoder frozen")
    
    def unfreeze_decoder(self):
        """Unfreeze decoder parameters"""
        for param in self.decoder.parameters():
            param.requires_grad = True
        logger.info("Decoder unfrozen")


def create_model_from_config(config, input_dim: int) -> ClaimsAutoencoder:
    """
    Create autoencoder model from configuration.
    
    Args:
        config: Configuration object
        input_dim: Number of input features
    
    Returns:
        Initialized autoencoder model
    """
    model_config = config.model
    
    model = ClaimsAutoencoder(
        input_dim=input_dim,
        encoding_dim=model_config.encoding_dim,
        hidden_layers=model_config.hidden_layers,
        dropout_rate=model_config.dropout_rate,
        batch_norm=model_config.batch_norm,
        activation=model_config.activation,
    )
    
    logger.info(f"Created model with {model.count_parameters():,} parameters")
    
    return model


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda x, y: x < (y - min_delta)
        else:
            self.monitor_op = lambda x, y: x > (y + min_delta)
    
    def __call__(self, current_score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current validation score
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = current_score
        elif self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
        
        return self.early_stop


if __name__ == "__main__":
    # Example usage
    from claims_fraud.config_manager import ConfigManager
    
    # Load config
    config_manager = ConfigManager("config/example_config.yaml")
    config = config_manager.get_config()
    
    # Create model
    input_dim = 20  # Example input dimension
    model = create_model_from_config(config, input_dim)
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    print(f"\nLayer sizes: {model.get_layer_sizes()}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    reconstruction, encoding = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Encoding shape: {encoding.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Test anomaly scoring
    anomaly_scores = model.get_anomaly_score(x)
    print(f"Anomaly scores shape: {anomaly_scores.shape}")
