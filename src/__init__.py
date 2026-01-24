"""
Claims Autoencoder Package
A production-ready anomaly detection system for insurance claims.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.config_manager import ConfigManager
from src.model_architecture import ClaimsAutoencoder

__all__ = [
    "ConfigManager",
    "ClaimsAutoencoder",
]
