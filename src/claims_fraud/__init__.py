"""
Claims Fraud Detection Package

A comprehensive fraud detection system for insurance claims using
tree-based models, fairness analysis, and drift monitoring.
"""

from .__version__ import __version__, __author__, __license__

# Core components (lazy imports to avoid circular dependencies)
def __getattr__(name):
    """Lazy import for better startup time"""
    if name == "FraudDetector":
        from .core.scoring import FraudDetector
        return FraudDetector
    elif name == "TreeModel":
        from .core.tree_models import ClaimsTreeAutoencoder
        return ClaimsTreeAutoencoder
    elif name == "FairnessAnalyzer":
        from .analysis.fairness import FairnessAnalyzer
        return FairnessAnalyzer
    elif name == "PSIMonitor":
        from .analysis.monitoring import PSIMonitor
        return PSIMonitor
    elif name == "DataPipeline":
        from .data.ingestion import DataIngestion
        return DataIngestion
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    '__version__',
    '__author__',
    '__license__',
    'FraudDetector',
    'TreeModel',
    'FairnessAnalyzer',
    'PSIMonitor',
    'DataPipeline',
]
