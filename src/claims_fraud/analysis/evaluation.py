"""
Evaluation Module for Claims Autoencoder
Handles model evaluation and metrics computation.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report
)
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluator for Claims Autoencoder.
    
    Computes:
    - Reconstruction errors
    - Anomaly scores
    - Precision/Recall at K
    - ROC curves
    - Visualization plots
    """
    
    def __init__(self, config, model, device='cpu'):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration object
            model: Trained autoencoder model
            device: Device to run evaluation on
        """
        self.config = config
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        self.anomaly_threshold = None
    
    def compute_reconstruction_errors(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Compute reconstruction errors for data.
        
        Args:
            X: Input data array [n_samples, n_features]
        
        Returns:
            Reconstruction errors [n_samples]
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            reconstruction, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstruction) ** 2, dim=1)
        
        return errors.cpu().numpy()
    
    def set_anomaly_threshold(
        self,
        X_train: np.ndarray,
        percentile: Optional[float] = None
    ):
        """
        Set anomaly threshold based on training data.
        
        Args:
            X_train: Training data
            percentile: Percentile for threshold (uses config if None)
        """
        if percentile is None:
            percentile = self.config.model.anomaly_threshold_percentile
        
        train_errors = self.compute_reconstruction_errors(X_train)
        self.anomaly_threshold = np.percentile(train_errors, percentile)
        
        logger.info(f"Anomaly threshold set to {self.anomaly_threshold:.6f} "
                   f"({percentile}th percentile)")
    
    def predict_anomalies(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies in data.
        
        Args:
            X: Input data
            threshold: Anomaly threshold (uses stored if None)
        
        Returns:
            Tuple of (anomaly_labels, anomaly_scores)
        """
        if threshold is None:
            if self.anomaly_threshold is None:
                raise ValueError("Anomaly threshold not set. Call set_anomaly_threshold first.")
            threshold = self.anomaly_threshold
        
        scores = self.compute_reconstruction_errors(X)
        labels = (scores > threshold).astype(int)
        
        return labels, scores
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test data
            y_test: True anomaly labels (optional)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Compute reconstruction errors
        errors = self.compute_reconstruction_errors(X_test)
        
        metrics['mean_reconstruction_error'] = float(np.mean(errors))
        metrics['std_reconstruction_error'] = float(np.std(errors))
        metrics['min_reconstruction_error'] = float(np.min(errors))
        metrics['max_reconstruction_error'] = float(np.max(errors))
        
        # If true labels provided, compute classification metrics
        if y_test is not None:
            predictions, scores = self.predict_anomalies(X_test)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, predictions)
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                metrics['true_negatives'] = int(tn)
                metrics['false_positives'] = int(fp)
                metrics['false_negatives'] = int(fn)
                metrics['true_positives'] = int(tp)
                
                # Derived metrics
                metrics['accuracy'] = float((tp + tn) / (tp + tn + fp + fn))
                metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                metrics['f1_score'] = float(
                    2 * metrics['precision'] * metrics['recall'] /
                    (metrics['precision'] + metrics['recall'])
                ) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
                
                # False positive rate
                metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            
            # Anomaly detection rate
            metrics['anomaly_detection_rate'] = float(np.mean(predictions))
            
            # Precision and Recall at K
            for k in self.config.evaluation.k_values:
                precision_k, recall_k = self.precision_recall_at_k(scores, y_test, k)
                metrics[f'precision_at_{k}'] = precision_k
                metrics[f'recall_at_{k}'] = recall_k
        
        logger.info("Evaluation completed")
        return metrics
    
    def precision_recall_at_k(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        k: int
    ) -> Tuple[float, float]:
        """
        Compute precision and recall at top K anomalies.
        
        Args:
            scores: Anomaly scores
            y_true: True labels
            k: Number of top anomalies to consider
        
        Returns:
            Tuple of (precision_at_k, recall_at_k)
        """
        # Get indices of top K scores
        top_k_indices = np.argsort(scores)[-k:]
        
        # Predictions: top K are anomalies
        predictions = np.zeros_like(y_true)
        predictions[top_k_indices] = 1
        
        # Compute precision and recall
        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return float(precision), float(recall)
    
    def plot_reconstruction_error_distribution(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of reconstruction errors.
        
        Args:
            X: Input data
            y: True labels (optional)
            save_path: Path to save plot
        """
        errors = self.compute_reconstruction_errors(X)
        
        plt.figure(figsize=(10, 6))
        
        if y is not None:
            # Separate normal and anomalous
            normal_errors = errors[y == 0]
            anomaly_errors = errors[y == 1]
            
            plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal', density=True)
            plt.hist(anomaly_errors, bins=50, alpha=0.6, label='Anomaly', density=True)
        else:
            plt.hist(errors, bins=50, alpha=0.6, density=True)
        
        if self.anomaly_threshold is not None:
            plt.axvline(self.anomaly_threshold, color='r', linestyle='--',
                       label=f'Threshold: {self.anomaly_threshold:.4f}')
        
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve.
        
        Args:
            X: Input data
            y: True labels
            save_path: Path to save plot
        """
        scores = self.compute_reconstruction_errors(X)
        
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return roc_auc
    
    def plot_precision_recall_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot precision-recall curve.
        
        Args:
            X: Input data
            y: True labels
            save_path: Path to save plot
        """
        scores = self.compute_reconstruction_errors(X)
        
        precision, recall, _ = precision_recall_curve(y, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_evaluation_report(
        self,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            X_test: Test data
            y_test: True labels (optional)
            output_dir: Directory to save plots
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.evaluate(X_test, y_test)
        
        if output_dir and self.config.evaluation.save_plots:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot reconstruction error distribution
            if self.config.evaluation.plot_distributions:
                self.plot_reconstruction_error_distribution(
                    X_test, y_test,
                    save_path=str(output_dir / 'reconstruction_error_dist.png')
                )
            
            # Plot ROC curve if labels available
            if y_test is not None and self.config.evaluation.plot_roc_curve:
                roc_auc = self.plot_roc_curve(
                    X_test, y_test,
                    save_path=str(output_dir / 'roc_curve.png')
                )
                metrics['roc_auc'] = roc_auc
                
                self.plot_precision_recall_curve(
                    X_test, y_test,
                    save_path=str(output_dir / 'precision_recall_curve.png')
                )
        
        return metrics


if __name__ == "__main__":
    # Example usage
    from ..config.manager import ConfigManager
    from src.model_architecture import create_model_from_config
    
    # Load config and model
    config_manager = ConfigManager("config/example_config.yaml")
    config = config_manager.get_config()
    
    # Create dummy model and data
    input_dim = 20
    model = create_model_from_config(config, input_dim)
    
    X_test = np.random.randn(1000, input_dim)
    y_test = np.random.randint(0, 2, 1000)
    
    # Evaluate
    evaluator = ModelEvaluator(config, model)
    evaluator.set_anomaly_threshold(X_test[:500])
    
    metrics = evaluator.generate_evaluation_report(X_test, y_test, "outputs/plots")
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
