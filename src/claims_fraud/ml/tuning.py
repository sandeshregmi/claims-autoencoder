"""
Hyperparameter Tuning Module
Uses Optuna for hyperparameter optimization.
"""

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
import torch
import numpy as np
from typing import Dict, Optional
import logging
from pathlib import Path

from ..config.manager import ConfigManager
from src.model_architecture import ClaimsAutoencoder
from .training import ClaimsTrainer


logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuning using Optuna.
    
    Optimizes:
    - Model architecture (encoding_dim, hidden_layers)
    - Training parameters (learning_rate, batch_size, dropout_rate)
    """
    
    def __init__(
        self,
        config,
        train_data: np.ndarray,
        val_data: np.ndarray,
        device: str = 'cpu'
    ):
        """
        Initialize tuner.
        
        Args:
            config: Configuration object
            train_data: Training data
            val_data: Validation data
            device: Device to use
        """
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.tuning_config = config.hyperparameter_tuning
        
        self.input_dim = train_data.shape[1]
        self.best_params = None
        self.best_value = None
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Validation loss
        """
        # Sample hyperparameters
        encoding_dim = trial.suggest_categorical(
            'encoding_dim',
            self.tuning_config.search_space.get('encoding_dim', [16, 32, 64])
        )
        
        # Sample hidden layers architecture
        hidden_layers_options = self.tuning_config.search_space.get(
            'hidden_layers',
            [[64], [128, 64], [256, 128, 64]]
        )
        hidden_layers_idx = trial.suggest_categorical(
            'hidden_layers',
            range(len(hidden_layers_options))
        )
        hidden_layers = hidden_layers_options[hidden_layers_idx]
        
        dropout_rate = trial.suggest_categorical(
            'dropout_rate',
            self.tuning_config.search_space.get('dropout_rate', [0.1, 0.2, 0.3])
        )
        
        learning_rate = trial.suggest_categorical(
            'learning_rate',
            self.tuning_config.search_space.get('learning_rate', [0.001, 0.01])
        )
        
        batch_size = trial.suggest_categorical(
            'batch_size',
            self.tuning_config.search_space.get('batch_size', [128, 256, 512])
        )
        
        # Create model with sampled hyperparameters
        model = ClaimsAutoencoder(
            input_dim=self.input_dim,
            encoding_dim=encoding_dim,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            batch_norm=self.config.model.batch_norm,
            activation=self.config.model.activation
        )
        
        # Update config with sampled parameters
        temp_config = self.config.copy(deep=True)
        temp_config.training.learning_rate = learning_rate
        temp_config.training.batch_size = batch_size
        temp_config.training.max_epochs = 20  # Shorter for tuning
        
        # Create data loaders
        from torch.utils.data import DataLoader, TensorDataset
        
        train_dataset = TensorDataset(torch.FloatTensor(self.train_data))
        val_dataset = TensorDataset(torch.FloatTensor(self.val_data))
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Train model
        trainer = ClaimsTrainer(temp_config, model, device=self.device)
        
        # Train for reduced epochs
        for epoch in range(1, 21):  # Max 20 epochs for tuning
            train_loss = trainer.train_epoch(train_loader, epoch)
            val_loss = trainer.validate(val_loader)
            
            # Report intermediate value for pruning
            trial.report(val_loss, epoch)
            
            # Prune if needed
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Return final validation loss
        return val_loss
    
    def optimize(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials (uses config if None)
            timeout: Timeout in seconds (uses config if None)
        
        Returns:
            Dictionary with best parameters
        """
        if n_trials is None:
            n_trials = self.tuning_config.n_trials
        
        if timeout is None:
            timeout = self.tuning_config.timeout
        
        # Create pruner
        pruner_name = self.tuning_config.pruner
        if pruner_name == 'median':
            pruner = MedianPruner()
        elif pruner_name == 'hyperband':
            pruner = HyperbandPruner()
        else:
            pruner = None
        
        # Create study
        direction = self.tuning_config.direction
        study = optuna.create_study(direction=direction, pruner=pruner)
        
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Store best results
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        logger.info(f"Best value: {self.best_value:.6f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def update_config_with_best_params(self) -> ConfigManager:
        """
        Update configuration with best parameters.
        
        Returns:
            Updated configuration
        """
        if self.best_params is None:
            raise ValueError("No optimization results. Run optimize() first.")
        
        # Update model config
        if 'encoding_dim' in self.best_params:
            self.config.model.encoding_dim = self.best_params['encoding_dim']
        
        if 'hidden_layers' in self.best_params:
            hidden_layers_options = self.tuning_config.search_space['hidden_layers']
            hidden_layers_idx = self.best_params['hidden_layers']
            self.config.model.hidden_layers = hidden_layers_options[hidden_layers_idx]
        
        if 'dropout_rate' in self.best_params:
            self.config.model.dropout_rate = self.best_params['dropout_rate']
        
        # Update training config
        if 'learning_rate' in self.best_params:
            self.config.training.learning_rate = self.best_params['learning_rate']
        
        if 'batch_size' in self.best_params:
            self.config.training.batch_size = self.best_params['batch_size']
        
        logger.info("Configuration updated with best parameters")
        
        return self.config


def tune_and_train():
    """
    Complete hyperparameter tuning and training pipeline.
    """
    import argparse
    from src.data_ingestion import DataIngestion
    from src.preprocessing import ClaimsPreprocessor
    
    parser = argparse.ArgumentParser(description='Tune and train autoencoder')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # Load and preprocess data
    logger.info("Loading data...")
    data_ingestion = DataIngestion(config)
    train_df, val_df, test_df = data_ingestion.load_train_val_test()
    
    logger.info("Preprocessing data...")
    preprocessor = ClaimsPreprocessor(config)
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    
    # Tune hyperparameters
    if config.hyperparameter_tuning.enabled:
        logger.info("Starting hyperparameter tuning...")
        tuner = HyperparameterTuner(config, X_train, X_val, device=args.device)
        best_params = tuner.optimize()
        
        # Update config
        config = tuner.update_config_with_best_params()
        
        # Save updated config
        config_manager.save_config("config/tuned_config.yaml")
        logger.info("Saved tuned configuration to config/tuned_config.yaml")
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters...")
    from src.model_architecture import create_model_from_config
    from torch.utils.data import DataLoader, TensorDataset
    
    model = create_model_from_config(config, X_train.shape[1])
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )
    
    trainer = ClaimsTrainer(config, model, device=args.device)
    trainer.train(train_loader, val_loader)
    
    # Save final model
    model_path = f"{config.paths.models_dir}/tuned_model.pth"
    trainer.save_model(model_path)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    tune_and_train()
