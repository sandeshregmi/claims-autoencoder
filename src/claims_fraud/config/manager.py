"""Configuration Manager - Enhanced Version 2.0"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class DictConfig:
    """Dictionary-based configuration with attribute and dict access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from dictionary."""
        self._data = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = DictConfig(value)
            elif isinstance(value, list):
                value = [DictConfig(v) if isinstance(v, dict) else v for v in value]
            self._data[key] = value
            setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        return self._data.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access."""
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any):
        """Allow dict-style assignment."""
        self._data[key] = value
        setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data
    
    def items(self):
        """Return items like a dictionary."""
        return self._data.items()
    
    def keys(self):
        """Return keys like a dictionary."""
        return self._data.keys()
    
    def values(self):
        """Return values like a dictionary."""
        return self._data.values()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, DictConfig):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [v.to_dict() if isinstance(v, DictConfig) else v for v in value]
            else:
                result[key] = value
        return result
    
    def __hash__(self):
        """Make hashable for Streamlit caching."""
        # Convert to JSON string and hash it
        json_str = json.dumps(self.to_dict(), sort_keys=True)
        return int(hashlib.md5(json_str.encode()).hexdigest(), 16)
    
    def __eq__(self, other):
        """Equality comparison for hashing."""
        if not isinstance(other, DictConfig):
            return False
        return self.to_dict() == other.to_dict()
    
    def __repr__(self):
        """String representation."""
        return f"DictConfig({self.to_dict()})"


class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: str):
        """Initialize configuration manager."""
        self.config_path = Path(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self._config = DictConfig(config_dict)
        logger.info(f"Configuration loaded from {self.config_path}")
    
    def get_config(self) -> DictConfig:
        """Get configuration object."""
        return self._config
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()
        logger.info("Configuration reloaded")


def load_config(config_path: str = "config/config.yaml") -> DictConfig:
    """Convenience function to load configuration."""
    manager = ConfigManager(config_path)
    return manager.get_config()
