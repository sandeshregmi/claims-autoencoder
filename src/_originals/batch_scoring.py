"""
Batch Scoring Module
Efficiently scores large batches of claims data.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import logging
from tqdm import tqdm
import argparse

from src.config_manager import ConfigManager
from src.preprocessing import ClaimsPreprocessor
from src.model_architecture import ClaimsAutoencoder


logger = logging.getLogger(__name__)


class BatchScorer:
    """
    Batch scoring for claims autoencoder.
    
    Features:
    - Efficient chunked processing
    - Memory-efficient streaming
    - Progress tracking
    - Multiple output formats
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        preprocessor: ClaimsPreprocessor,
        config,
        device: str = 'cpu'
    ):
        """
        Initialize batch scorer.
        
        Args:
            model: Trained autoencoder model
            preprocessor: Fitted preprocessor
            config: Configuration object
            device: Device to use
        """
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.chunk_size = config.batch_scoring.chunk_size
    
    def score_batch(self, X: np.ndarray) -> dict:
        """
        Score a batch of data.
        
        Args:
            X: Input data [n_samples, n_features]
        
        Returns:
            Dictionary with scores and predictions
        """
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Get reconstruction and encoding
            reconstruction, encoding = self.model(X_tensor)
            
            # Compute reconstruction error
            errors = torch.mean((X_tensor - reconstruction) ** 2, dim=1)
            
            results = {
                'reconstruction_error': errors.cpu().numpy(),
                'encoding': encoding.cpu().numpy(),
            }
            
            if self.config.batch_scoring.save_reconstructions:
                results['reconstruction'] = reconstruction.cpu().numpy()
            
            return results
    
    def score_dataframe(
        self,
        df: pd.DataFrame,
        anomaly_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Score a DataFrame of claims.
        
        Args:
            df: Input DataFrame with claim features
            anomaly_threshold: Threshold for anomaly detection
        
        Returns:
            DataFrame with scores added
        """
        logger.info(f"Scoring {len(df)} claims...")
        
        # Preprocess data
        X = self.preprocessor.transform(df)
        
        # Score in chunks
        all_errors = []
        all_encodings = []
        all_reconstructions = []
        
        num_chunks = (len(X) + self.chunk_size - 1) // self.chunk_size
        
        for i in tqdm(range(num_chunks), desc="Scoring batches"):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(X))
            
            X_chunk = X[start_idx:end_idx]
            results = self.score_batch(X_chunk)
            
            all_errors.append(results['reconstruction_error'])
            all_encodings.append(results['encoding'])
            
            if 'reconstruction' in results:
                all_reconstructions.append(results['reconstruction'])
        
        # Concatenate results
        reconstruction_errors = np.concatenate(all_errors)
        encodings = np.concatenate(all_encodings)
        
        # Create output DataFrame
        output_df = df.copy()
        output_df['reconstruction_error'] = reconstruction_errors
        
        # Add anomaly predictions if threshold provided
        if anomaly_threshold is not None:
            output_df['is_anomaly'] = (reconstruction_errors > anomaly_threshold).astype(int)
            output_df['anomaly_score'] = reconstruction_errors / anomaly_threshold
        
        # Add encoding features
        for i in range(encodings.shape[1]):
            output_df[f'encoding_{i}'] = encodings[:, i]
        
        # Add reconstructions if saved
        if all_reconstructions:
            reconstructions = np.concatenate(all_reconstructions)
            for i in range(reconstructions.shape[1]):
                output_df[f'reconstruction_{i}'] = reconstructions[:, i]
        
        logger.info("Scoring completed")
        return output_df
    
    def score_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        anomaly_threshold: Optional[float] = None,
        output_format: Optional[str] = None
    ):
        """
        Score claims from a file and save results.
        
        Args:
            input_path: Path to input file
            output_path: Path to save results
            anomaly_threshold: Threshold for anomaly detection
            output_format: Output format ('parquet', 'csv')
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if output_format is None:
            output_format = self.config.batch_scoring.output_format
        
        # Load data
        logger.info(f"Loading data from {input_path}")
        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported input format: {input_path.suffix}")
        
        # Score data
        scored_df = self.score_dataframe(df, anomaly_threshold)
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_path}")
        if output_format == 'parquet':
            scored_df.to_parquet(output_path, index=False)
        elif output_format == 'csv':
            scored_df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info("Scoring completed successfully")
    
    def get_top_anomalies(
        self,
        df: pd.DataFrame,
        n: int = 100
    ) -> pd.DataFrame:
        """
        Get top N anomalies from scored data.
        
        Args:
            df: Scored DataFrame
            n: Number of top anomalies to return
        
        Returns:
            DataFrame with top anomalies
        """
        if 'reconstruction_error' not in df.columns:
            raise ValueError("DataFrame must be scored first")
        
        return df.nlargest(n, 'reconstruction_error')
    
    def generate_scoring_report(
        self,
        df: pd.DataFrame,
        anomaly_threshold: Optional[float] = None
    ) -> dict:
        """
        Generate summary report of scoring results.
        
        Args:
            df: Scored DataFrame
            anomaly_threshold: Anomaly threshold
        
        Returns:
            Dictionary with summary statistics
        """
        if 'reconstruction_error' not in df.columns:
            raise ValueError("DataFrame must be scored first")
        
        errors = df['reconstruction_error']
        
        report = {
            'total_claims': len(df),
            'mean_reconstruction_error': float(errors.mean()),
            'std_reconstruction_error': float(errors.std()),
            'min_reconstruction_error': float(errors.min()),
            'max_reconstruction_error': float(errors.max()),
            'median_reconstruction_error': float(errors.median()),
            'q25_reconstruction_error': float(errors.quantile(0.25)),
            'q75_reconstruction_error': float(errors.quantile(0.75)),
            'q95_reconstruction_error': float(errors.quantile(0.95)),
            'q99_reconstruction_error': float(errors.quantile(0.99)),
        }
        
        if anomaly_threshold is not None:
            anomalies = df[df['reconstruction_error'] > anomaly_threshold]
            report['anomaly_threshold'] = anomaly_threshold
            report['num_anomalies'] = len(anomalies)
            report['anomaly_rate'] = len(anomalies) / len(df) * 100
        
        return report


def main():
    """Main batch scoring script"""
    parser = argparse.ArgumentParser(description='Batch score claims data')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--preprocessor-path', type=str, required=True,
                       help='Path to fitted preprocessor')
    parser.add_argument('--input-path', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to save scored data')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Anomaly threshold')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cuda, cpu, mps)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # Load preprocessor
    logger.info("Loading preprocessor...")
    preprocessor = ClaimsPreprocessor.load(args.preprocessor_path, config)
    
    # Load model
    logger.info("Loading model...")
    model = torch.load(args.model_path, map_location=args.device)
    
    # Create scorer
    scorer = BatchScorer(model, preprocessor, config, device=args.device)
    
    # Score data
    scorer.score_file(
        input_path=args.input_path,
        output_path=args.output_path,
        anomaly_threshold=args.threshold
    )
    
    # Generate and display report
    df = pd.read_parquet(args.output_path)
    report = scorer.generate_scoring_report(df, args.threshold)
    
    print("\n" + "="*50)
    print("BATCH SCORING REPORT")
    print("="*50)
    for key, value in report.items():
        print(f"{key}: {value}")
    print("="*50)


if __name__ == "__main__":
    main()
