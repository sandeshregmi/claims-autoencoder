"""
Helper module for loading config from any directory
Use this in Jupyter notebooks
"""

from pathlib import Path
import sys

# Add src to path if not already there
project_root = Path('/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder')
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from claims_fraud.config.manager import load_config as _load_config, ConfigManager, DictConfig

def load_config(config_path=None):
    """
    Load config from any directory.
    
    Args:
        config_path: Path to config file (optional)
                    If None, uses default project config
    
    Returns:
        Config object
    
    Example:
        from claims_fraud.notebook_helper import load_config
        config = load_config()
    """
    if config_path is None:
        # Use default project config
        config_path = project_root / 'config' / 'config.yaml'
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            # Make relative paths absolute from project root
            config_path = project_root / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Project root: {project_root}\n"
            f"Make sure you're in the right directory or provide full path."
        )
    
    return _load_config(str(config_path))


# Make it easy to import
__all__ = ['load_config', 'ConfigManager', 'DictConfig', 'project_root']
