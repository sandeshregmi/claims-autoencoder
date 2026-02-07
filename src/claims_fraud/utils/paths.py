"""Path management utilities"""
from pathlib import Path
import os

class PathManager:
    """Manage project paths"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or os.getcwd())
    
    @property
    def data_dir(self) -> Path:
        """Get data directory"""
        path = self.base_path / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def models_dir(self) -> Path:
        """Get models directory"""
        path = self.base_path / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def config_dir(self) -> Path:
        """Get config directory"""
        return self.base_path / "configs"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory"""
        path = self.base_path / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def results_dir(self) -> Path:
        """Get results directory"""
        path = self.base_path / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path
