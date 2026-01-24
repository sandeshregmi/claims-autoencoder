"""
Configuration Manager for Claims Autoencoder System
Handles loading, validation, and management of configuration files.
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    """Data configuration settings"""
    train_path: str
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    numerical_features: List[str]
    categorical_features: List[str]
    handle_missing: str = "median"
    outlier_treatment: Dict[str, Any] = Field(default_factory=dict)
    feature_interactions: Dict[str, Any] = Field(default_factory=dict)
    split_ratios: Dict[str, float] = Field(default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15})
    
    @validator("handle_missing")
    def validate_missing_strategy(cls, v):
        valid = ["median", "mean", "drop", "forward_fill"]
        if v not in valid:
            raise ValueError(f"handle_missing must be one of {valid}")
        return v
    
    @validator("split_ratios")
    def validate_split_ratios(cls, v):
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return v


class ModelConfig(BaseModel):
    """Model architecture configuration"""
    encoding_dim: int = Field(gt=0)
    hidden_layers: List[int]
    activation: str = "relu"
    dropout_rate: float = Field(ge=0.0, le=1.0)
    batch_norm: bool = True
    anomaly_threshold_percentile: float = Field(ge=0.0, le=100.0, default=95.0)
    
    @validator("activation")
    def validate_activation(cls, v):
        valid = ["relu", "tanh", "leaky_relu", "elu", "selu"]
        if v not in valid:
            raise ValueError(f"activation must be one of {valid}")
        return v
    
    @validator("hidden_layers")
    def validate_hidden_layers(cls, v):
        if not v:
            raise ValueError("hidden_layers cannot be empty")
        if any(x <= 0 for x in v):
            raise ValueError("All hidden layer sizes must be positive")
        return v


class TrainingConfig(BaseModel):
    """Training configuration settings"""
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0, default=0.0001)
    optimizer: str = "adam"
    lr_scheduler: Dict[str, Any] = Field(default_factory=dict)
    max_epochs: int = Field(gt=0)
    early_stopping: Dict[str, Any] = Field(default_factory=dict)
    gradient_clip_val: float = Field(ge=0.0, default=1.0)
    seed: int = 42
    deterministic: bool = True
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
    
    @validator("optimizer")
    def validate_optimizer(cls, v):
        valid = ["adam", "sgd", "adamw", "rmsprop"]
        if v not in valid:
            raise ValueError(f"optimizer must be one of {valid}")
        return v
    
    @validator("precision")
    def validate_precision(cls, v):
        valid = ["32", "16-mixed", "bf16-mixed", "64"]
        if v not in valid:
            raise ValueError(f"precision must be one of {valid}")
        return v


class MLflowConfig(BaseModel):
    """MLflow tracking configuration"""
    enabled: bool = True
    tracking_uri: str = "mlruns"
    experiment_name: str = "claims_autoencoder"
    run_name: Optional[str] = None
    log_params: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True
    log_model: bool = True
    tags: Dict[str, str] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""
    metrics: List[str]
    k_values: List[int] = Field(default_factory=lambda: [10, 50, 100])
    plot_distributions: bool = True
    plot_roc_curve: bool = True
    save_plots: bool = True
    plots_dir: str = "outputs/plots"


class MonitoringConfig(BaseModel):
    """Monitoring and drift detection configuration"""
    psi: Dict[str, Any] = Field(default_factory=dict)
    drift_detection: Dict[str, Any] = Field(default_factory=dict)


class BatchScoringConfig(BaseModel):
    """Batch scoring configuration"""
    chunk_size: int = Field(gt=0, default=10000)
    num_workers: int = Field(ge=0, default=4)
    save_reconstructions: bool = False
    output_format: str = "parquet"
    
    @validator("output_format")
    def validate_output_format(cls, v):
        valid = ["parquet", "csv", "json"]
        if v not in valid:
            raise ValueError(f"output_format must be one of {valid}")
        return v


class HyperparameterTuningConfig(BaseModel):
    """Hyperparameter tuning configuration"""
    enabled: bool = False
    n_trials: int = Field(gt=0, default=50)
    timeout: Optional[int] = Field(ge=0, default=3600)
    search_space: Dict[str, Any] = Field(default_factory=dict)
    direction: str = "minimize"
    metric: str = "val_loss"
    pruner: Optional[str] = "median"
    
    @validator("direction")
    def validate_direction(cls, v):
        valid = ["minimize", "maximize"]
        if v not in valid:
            raise ValueError(f"direction must be one of {valid}")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    log_file: Optional[str] = "logs/training.log"
    log_to_console: bool = True
    log_to_file: bool = True
    
    @validator("level")
    def validate_level(cls, v):
        valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid:
            raise ValueError(f"level must be one of {valid}")
        return v


class PathsConfig(BaseModel):
    """Paths configuration"""
    data_dir: str = "data"
    models_dir: str = "models"
    outputs_dir: str = "outputs"
    logs_dir: str = "logs"
    checkpoints_dir: str = "checkpoints"


class FeatureStoreConfig(BaseModel):
    """Feature store configuration"""
    enabled: bool = False
    backend: str = "local"
    cache_ttl: int = Field(ge=0, default=3600)


class Config(BaseModel):
    """Main configuration class"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    evaluation: EvaluationConfig
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    batch_scoring: BatchScoringConfig = Field(default_factory=BatchScoringConfig)
    hyperparameter_tuning: HyperparameterTuningConfig = Field(default_factory=HyperparameterTuningConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    
    class Config:
        arbitrary_types_allowed = True


class ConfigManager:
    """
    Manages configuration loading, validation, and access.
    
    Features:
    - YAML configuration file loading
    - Pydantic-based validation
    - Default value handling
    - Environment variable override support
    - Configuration merging
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[Config] = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> Config:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validated configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
            ValidationError: If configuration is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Validate and create config object
        self.config = Config(**config_dict)
        self.config_path = config_path
        
        logger.info("Configuration loaded and validated successfully")
        return self.config
    
    def save_config(self, output_path: Union[str, Path]):
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        if self.config is None:
            raise ValueError("No configuration loaded")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.config.dict()
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def get_config(self) -> Config:
        """
        Get current configuration.
        
        Returns:
            Configuration object
            
        Raises:
            ValueError: If no configuration is loaded
        """
        if self.config is None:
            raise ValueError("No configuration loaded. Call load_config() first.")
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        if self.config is None:
            raise ValueError("No configuration loaded")
        
        config_dict = self.config.dict()
        config_dict = self._deep_update(config_dict, updates)
        self.config = Config(**config_dict)
        
        logger.info("Configuration updated")
    
    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
        """
        Recursively update nested dictionary.
        
        Args:
            base_dict: Base dictionary
            update_dict: Updates to apply
            
        Returns:
            Updated dictionary
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                base_dict[key] = ConfigManager._deep_update(
                    base_dict.get(key, {}), value
                )
            else:
                base_dict[key] = value
        return base_dict
    
    def validate_paths(self):
        """
        Validate that all configured paths exist or can be created.
        Creates directories if they don't exist.
        """
        if self.config is None:
            raise ValueError("No configuration loaded")
        
        paths = self.config.paths
        dirs_to_create = [
            paths.data_dir,
            paths.models_dir,
            paths.outputs_dir,
            paths.logs_dir,
            paths.checkpoints_dir,
        ]
        
        for dir_path in dirs_to_create:
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
        
        # Validate data files exist
        data_config = self.config.data
        if data_config.train_path and not Path(data_config.train_path).exists():
            logger.warning(f"Training data not found: {data_config.train_path}")
    
    def get_all_features(self) -> List[str]:
        """
        Get list of all features (numerical + categorical).
        
        Returns:
            List of all feature names
        """
        if self.config is None:
            raise ValueError("No configuration loaded")
        
        return (
            self.config.data.numerical_features +
            self.config.data.categorical_features
        )
    
    def get_feature_count(self) -> int:
        """
        Get total number of features.
        
        Returns:
            Total feature count
        """
        return len(self.get_all_features())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        if self.config is None:
            raise ValueError("No configuration loaded")
        return self.config.dict()
    
    def __repr__(self) -> str:
        """String representation"""
        if self.config is None:
            return "ConfigManager(no config loaded)"
        return f"ConfigManager(config_path={self.config_path})"


def create_default_config() -> Config:
    """
    Create a default configuration object.
    
    Returns:
        Default configuration
    """
    default_config = {
        "data": {
            "train_path": "data/claims_train.parquet",
            "numerical_features": ["claim_amount", "patient_age"],
            "categorical_features": ["claim_type"],
            "handle_missing": "median",
        },
        "model": {
            "encoding_dim": 32,
            "hidden_layers": [128, 64],
            "dropout_rate": 0.3,
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 0.001,
            "max_epochs": 100,
        },
        "evaluation": {
            "metrics": ["reconstruction_error"],
        },
    }
    
    return Config(**default_config)


if __name__ == "__main__":
    # Example usage
    manager = ConfigManager()
    
    # Load from file
    config = manager.load_config("config/example_config.yaml")
    
    print(f"Loaded config with {manager.get_feature_count()} features")
    print(f"Model encoding dimension: {config.model.encoding_dim}")
    print(f"Training batch size: {config.training.batch_size}")
