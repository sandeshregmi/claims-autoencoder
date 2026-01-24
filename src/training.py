"""
Training Module for Claims Autoencoder
Handles model training with MLflow integration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, Tuple
import logging
from pathlib import Path
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from src.config_manager import ConfigManager
from src.data_ingestion import DataIngestion
from src.preprocessing import ClaimsPreprocessor
from src.model_architecture import create_model_from_config, EarlyStopping


logger = logging.getLogger(__name__)


class ClaimsTrainer:
    """
    Trainer class for Claims Autoencoder.
    
    Features:
    - Training loop with validation
    - MLflow experiment tracking
    - Checkpointing
    - Early stopping
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        config,
        model: nn.Module,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            model: Autoencoder model
            device: Device to train on ('cpu', 'cuda', 'mps')
        """
        self.config = config
        self.model = model
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize early stopping
        self.early_stopping = self._create_early_stopping()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
        }
    
    def _create_optimizer(self) -> optim.Optimizer:
        """
        Create optimizer from config.
        
        Returns:
            Optimizer instance
        """
        training_config = self.config.training
        
        optimizer_name = training_config.optimizer.lower()
        lr = training_config.learning_rate
        weight_decay = training_config.weight_decay
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler from config.
        
        Returns:
            Scheduler instance or None
        """
        lr_config = self.config.training.lr_scheduler
        
        if not lr_config.get('enabled', False):
            return None
        
        scheduler_type = lr_config.get('type', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=lr_config.get('factor', 0.5),
                patience=lr_config.get('patience', 10),
                min_lr=lr_config.get('min_lr', 1e-6),
                verbose=True
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=lr_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=lr_config.get('step_size', 30),
                gamma=lr_config.get('gamma', 0.1)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _create_early_stopping(self) -> Optional[EarlyStopping]:
        """
        Create early stopping from config.
        
        Returns:
            EarlyStopping instance or None
        """
        es_config = self.config.training.early_stopping
        
        if not es_config.get('enabled', False):
            return None
        
        return EarlyStopping(
            patience=es_config.get('patience', 15),
            min_delta=es_config.get('min_delta', 0.001),
            mode='min'
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            if isinstance(batch, list):
                batch = batch[0]
            
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstruction, _ = self.model(batch)
            
            # Compute loss (MSE)
            loss = nn.functional.mse_loss(reconstruction, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_val
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, list):
                    batch = batch[0]
                
                batch = batch.to(self.device)
                
                # Forward pass
                reconstruction, _ = self.model(batch)
                
                # Compute loss
                loss = nn.functional.mse_loss(reconstruction, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ) -> Dict[str, list]:
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config if None)
        
        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.config.training.max_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log to console
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}"
            )
            
            # Log to MLflow
            if self.config.mlflow.enabled and self.config.mlflow.log_metrics:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=epoch)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
        
        logger.info("Training completed")
        return self.history
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Current validation loss
        """
        checkpoint_dir = Path(self.config.paths.checkpoints_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_model(self, path: str):
        """
        Save model weights.
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Claims Autoencoder')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, mps)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # Set random seed
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    
    # Load data
    logger.info("Loading data...")
    data_ingestion = DataIngestion(config)
    train_df, val_df, test_df = data_ingestion.load_train_val_test()
    
    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = ClaimsPreprocessor(config)
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    
    # Save preprocessor
    preprocessor.save(f"{config.paths.models_dir}/preprocessor.pkl")
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    logger.info("Creating model...")
    input_dim = X_train.shape[1]
    model = create_model_from_config(config, input_dim)
    
    # Initialize MLflow
    if config.mlflow.enabled:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        
        with mlflow.start_run(run_name=config.mlflow.run_name):
            # Log parameters
            if config.mlflow.log_params:
                mlflow.log_params({
                    'input_dim': input_dim,
                    'encoding_dim': config.model.encoding_dim,
                    'hidden_layers': str(config.model.hidden_layers),
                    'dropout_rate': config.model.dropout_rate,
                    'batch_size': config.training.batch_size,
                    'learning_rate': config.training.learning_rate,
                    'optimizer': config.training.optimizer,
                })
            
            # Create trainer and train
            trainer = ClaimsTrainer(config, model, device=args.device)
            history = trainer.train(train_loader, val_loader)
            
            # Save final model
            model_path = f"{config.paths.models_dir}/best_model.pth"
            trainer.save_model(model_path)
            
            # Log model to MLflow
            if config.mlflow.log_model:
                mlflow.pytorch.log_model(model, "model")
            
            logger.info("Training completed successfully!")
    
    else:
        # Train without MLflow
        trainer = ClaimsTrainer(config, model, device=args.device)
        history = trainer.train(train_loader, val_loader)
        
        # Save final model
        model_path = f"{config.paths.models_dir}/best_model.pth"
        trainer.save_model(model_path)
        
        logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
