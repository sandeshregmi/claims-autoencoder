"""
Data Ingestion Module for Claims Autoencoder
Handles loading and initial processing of claims data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List
import logging
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Data ingestion utilities for loading claims data from various formats.
    
    Supported formats:
    - Parquet
    - CSV
    - Pickle
    """
    
    def __init__(self, config):
        """
        Initialize data ingestion.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_config = config.data
    
    def load_data(
        self,
        file_path: Union[str, Path],
        file_format: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            file_path: Path to data file
            file_format: File format ('parquet', 'csv', 'pickle'). 
                        Auto-detected if None.
        
        Returns:
            DataFrame with loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Auto-detect format if not provided
        if file_format is None:
            file_format = file_path.suffix.lstrip('.')
        
        logger.info(f"Loading data from {file_path} (format: {file_format})")
        
        if file_format == 'parquet':
            df = pd.read_parquet(file_path)
        elif file_format == 'csv':
            df = pd.read_csv(file_path)
        elif file_format in ['pkl', 'pickle']:
            df = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def load_train_val_test(
        self
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load training, validation, and test datasets.
        
        If only train_path is provided, splits data according to split_ratios.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Load training data
        train_df = self.load_data(self.data_config.train_path)
        
        # Load validation and test if provided
        val_df = None
        test_df = None
        
        if self.data_config.val_path:
            val_df = self.load_data(self.data_config.val_path)
        
        if self.data_config.test_path:
            test_df = self.load_data(self.data_config.test_path)
        
        # If validation or test not provided, split from training data
        if val_df is None or test_df is None:
            logger.info("Splitting training data into train/val/test sets")
            train_df, val_df, test_df = self._split_data(train_df)
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df) if val_df is not None else 0}, "
                   f"Test: {len(test_df) if test_df is not None else 0}")
        
        return train_df, val_df, test_df
    
    def _split_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        split_ratios = self.data_config.split_ratios
        train_ratio = split_ratios['train']
        val_ratio = split_ratios['val']
        test_ratio = split_ratios['test']
        
        # Ensure ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=self.config.training.seed
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=self.config.training.seed
        )
        
        return train_df, val_df, test_df
    
    def validate_features(self, df: pd.DataFrame) -> None:
        """
        Validate that required features exist in DataFrame.
        
        Args:
            df: DataFrame to validate
        
        Raises:
            ValueError: If required features are missing
        """
        required_features = (
            self.data_config.numerical_features +
            self.data_config.categorical_features
        )
        
        missing_features = [f for f in required_features if f not in df.columns]
        
        if missing_features:
            raise ValueError(
                f"Missing required features in data: {missing_features}"
            )
        
        logger.debug("All required features present in data")
    
    def get_feature_types(self, df: pd.DataFrame) -> dict:
        """
        Get data types for all features.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dictionary mapping feature names to data types
        """
        feature_types = {}
        
        for col in self.data_config.numerical_features:
            if col in df.columns:
                feature_types[col] = 'numerical'
        
        for col in self.data_config.categorical_features:
            if col in df.columns:
                feature_types[col] = 'categorical'
        
        return feature_types
    
    def generate_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate summary statistics for the dataset.
        
        Args:
            df: DataFrame to summarize
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'numerical_features': {},
            'categorical_features': {},
            'missing_values': {},
        }
        
        # Numerical features
        for col in self.data_config.numerical_features:
            if col in df.columns:
                summary['numerical_features'][col] = {
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None,
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'missing_count': int(df[col].isna().sum()),
                    'missing_pct': float(df[col].isna().sum() / len(df) * 100),
                }
        
        # Categorical features
        for col in self.data_config.categorical_features:
            if col in df.columns:
                summary['categorical_features'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'top_value': str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None,
                    'missing_count': int(df[col].isna().sum()),
                    'missing_pct': float(df[col].isna().sum() / len(df) * 100),
                }
        
        # Overall missing values
        summary['missing_values'] = {
            'total_missing': int(df.isna().sum().sum()),
            'columns_with_missing': list(df.columns[df.isna().any()].tolist()),
        }
        
        return summary
    
    def sample_data(
        self,
        df: pd.DataFrame,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Sample data from DataFrame.
        
        Args:
            df: DataFrame to sample from
            n: Number of samples to return
            frac: Fraction of samples to return
            random_state: Random seed
        
        Returns:
            Sampled DataFrame
        """
        if random_state is None:
            random_state = self.config.training.seed
        
        if n is not None:
            return df.sample(n=min(n, len(df)), random_state=random_state)
        elif frac is not None:
            return df.sample(frac=frac, random_state=random_state)
        else:
            raise ValueError("Either n or frac must be specified")
    
    def filter_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Filter outliers from numerical columns.
        
        Args:
            df: DataFrame to filter
            columns: Columns to check for outliers (None = all numerical)
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame with outliers removed
        """
        if columns is None:
            columns = self.data_config.numerical_features
        
        df_filtered = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                z_scores = np.abs((df[col] - mean) / std)
                mask = z_scores <= threshold
            
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            rows_before = len(df_filtered)
            df_filtered = df_filtered[mask]
            rows_removed = rows_before - len(df_filtered)
            
            if rows_removed > 0:
                logger.info(f"Removed {rows_removed} outliers from {col}")
        
        return df_filtered
    
    def save_data(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        file_format: Optional[str] = None
    ):
        """
        Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            file_format: File format ('parquet', 'csv', 'pickle')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_format is None:
            file_format = file_path.suffix.lstrip('.')
        
        logger.info(f"Saving data to {file_path} (format: {file_format})")
        
        if file_format == 'parquet':
            df.to_parquet(file_path, index=False)
        elif file_format == 'csv':
            df.to_csv(file_path, index=False)
        elif file_format in ['pkl', 'pickle']:
            df.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Saved {len(df)} rows to {file_path}")


def load_sample_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate sample claims data for testing.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed
    
    Returns:
        DataFrame with sample claims data
    """
    np.random.seed(random_state)
    
    data = {
        'claim_amount': np.random.gamma(2, 1000, n_samples),
        'patient_age': np.random.randint(18, 90, n_samples),
        'provider_experience_years': np.random.randint(0, 40, n_samples),
        'days_since_last_claim': np.random.exponential(30, n_samples),
        'num_previous_claims': np.random.poisson(2, n_samples),
        'average_claim_amount': np.random.gamma(2, 800, n_samples),
        'claim_duration_days': np.random.randint(1, 60, n_samples),
        'claim_type': np.random.choice(['medical', 'dental', 'vision', 'prescription'], n_samples),
        'provider_specialty': np.random.choice(['general', 'specialist', 'surgeon', 'pediatrics'], n_samples),
        'diagnosis_code': np.random.choice(['D001', 'D002', 'D003', 'D004', 'D005'], n_samples),
        'procedure_code': np.random.choice(['P001', 'P002', 'P003', 'P004'], n_samples),
        'patient_gender': np.random.choice(['M', 'F', 'O'], n_samples),
        'geographic_region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    for col in df.columns:
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
    
    return df


if __name__ == "__main__":
    # Example usage
    from src.config_manager import ConfigManager
    
    # Load config
    config_manager = ConfigManager("config/example_config.yaml")
    config = config_manager.get_config()
    
    # Create sample data
    df = load_sample_data(n_samples=1000)
    df.to_parquet("data/sample_claims.parquet")
    
    # Load data using DataIngestion
    ingestion = DataIngestion(config)
    loaded_df = ingestion.load_data("data/sample_claims.parquet")
    
    # Generate summary
    summary = ingestion.generate_data_summary(loaded_df)
    print("Data Summary:", summary)
