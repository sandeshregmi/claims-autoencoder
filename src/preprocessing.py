"""
Preprocessing Module for Claims Autoencoder
Handles feature engineering, scaling, and encoding.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import logging
import pickle
from pathlib import Path


logger = logging.getLogger(__name__)


class ClaimsPreprocessor:
    """
    Preprocessing pipeline for claims data.
    
    Features:
    - Missing value imputation
    - Outlier treatment
    - Feature scaling
    - Categorical encoding
    - Feature interactions
    - Feature selection
    """
    
    def __init__(self, config):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_config = config.data
        
        # Initialize transformers
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.label_encoders = {}
        
        # Store feature statistics
        self.feature_stats = {}
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'ClaimsPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            df: Training DataFrame
        
        Returns:
            Self for method chaining
        """
        logger.info("Fitting preprocessor on training data")
        
        # Store original feature names
        self.numerical_features = self.data_config.numerical_features
        self.categorical_features = self.data_config.categorical_features
        
        # Fit numerical imputer
        if self.numerical_features:
            strategy = self._get_imputation_strategy()
            self.numerical_imputer = SimpleImputer(strategy=strategy)
            self.numerical_imputer.fit(df[self.numerical_features])
        
        # Fit categorical imputer
        if self.categorical_features:
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            self.categorical_imputer.fit(df[self.categorical_features])
        
        # Fit label encoders for categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            # Handle missing values before encoding
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                le.fit(non_null_values)
                self.label_encoders[col] = le
        
        # Fit scaler on imputed and encoded data
        df_processed = self._impute_and_encode(df)
        
        # Apply feature interactions if configured (need to do this before fitting scaler)
        if self.data_config.feature_interactions.get('enabled', False):
            df_processed = self._create_feature_interactions(df_processed)
        
        # Use RobustScaler for better handling of outliers
        self.scaler = RobustScaler()
        self.scaler.fit(df_processed)
        
        # Store feature statistics
        self._compute_feature_stats(df)
        
        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
        
        Returns:
            Transformed numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Impute and encode
        df_processed = self._impute_and_encode(df)
        
        # Apply feature interactions if configured
        if self.data_config.feature_interactions.get('enabled', False):
            df_processed = self._create_feature_interactions(df_processed)
        
        # Scale features
        X_scaled = self.scaler.transform(df_processed)
        
        return X_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessor and transform data.
        
        Args:
            df: DataFrame to fit and transform
        
        Returns:
            Transformed numpy array
        """
        self.fit(df)
        return self.transform(df)
    
    def _impute_and_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values and encode categorical features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Processed DataFrame
        """
        df_processed = pd.DataFrame()
        
        # Handle numerical features
        if self.numerical_features:
            numerical_imputed = self.numerical_imputer.transform(
                df[self.numerical_features]
            )
            numerical_df = pd.DataFrame(
                numerical_imputed,
                columns=self.numerical_features,
                index=df.index
            )
            df_processed = pd.concat([df_processed, numerical_df], axis=1)
        
        # Handle categorical features
        if self.categorical_features:
            categorical_imputed = self.categorical_imputer.transform(
                df[self.categorical_features]
            )
            
            # Encode categorical features
            for idx, col in enumerate(self.categorical_features):
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    col_values = categorical_imputed[:, idx]
                    
                    # Handle unseen categories
                    encoded_values = []
                    for val in col_values:
                        if val in le.classes_:
                            encoded_values.append(le.transform([val])[0])
                        else:
                            # Assign to most frequent class for unseen categories
                            encoded_values.append(le.transform([le.classes_[0]])[0])
                    
                    df_processed[col] = encoded_values
        
        return df_processed
    
    def _create_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with interaction features
        """
        pairs = self.data_config.feature_interactions.get('pairs', [])
        
        for feat1, feat2 in pairs:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                df[interaction_name] = df[feat1] * df[feat2]
                logger.debug(f"Created interaction feature: {interaction_name}")
        
        return df
    
    def _get_imputation_strategy(self) -> str:
        """
        Get imputation strategy from config.
        
        Returns:
            Strategy name for SimpleImputer
        """
        strategy_map = {
            'median': 'median',
            'mean': 'mean',
            'most_frequent': 'most_frequent',
        }
        
        config_strategy = self.data_config.handle_missing
        return strategy_map.get(config_strategy, 'median')
    
    def _compute_feature_stats(self, df: pd.DataFrame):
        """
        Compute and store feature statistics.
        
        Args:
            df: DataFrame to analyze
        """
        self.feature_stats = {}
        
        for col in self.numerical_features:
            if col in df.columns:
                self.feature_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75)),
                }
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'clip',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Handle outliers in numerical features.
        
        Args:
            df: Input DataFrame
            method: Method for handling outliers ('clip', 'remove', 'winsorize')
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame with outliers handled
        """
        if not self.data_config.outlier_treatment.get('enabled', False):
            return df
        
        df_processed = df.copy()
        
        for col in self.numerical_features:
            if col not in df.columns:
                continue
            
            if method == 'clip':
                # Clip to Q1 - threshold*IQR and Q3 + threshold*IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_processed[col] = df[col].clip(lower_bound, upper_bound)
            
            elif method == 'remove':
                # Remove rows with outliers (handled in data_ingestion)
                pass
            
            elif method == 'winsorize':
                # Winsorize to 1st and 99th percentiles
                lower_bound = df[col].quantile(0.01)
                upper_bound = df[col].quantile(0.99)
                df_processed[col] = df[col].clip(lower_bound, upper_bound)
        
        return df_processed
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original space.
        
        Args:
            X: Scaled data array
        
        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        
        return self.scaler.inverse_transform(X)
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        feature_names = self.numerical_features + self.categorical_features
        
        # Add interaction features if enabled
        if self.data_config.feature_interactions.get('enabled', False):
            pairs = self.data_config.feature_interactions.get('pairs', [])
            for feat1, feat2 in pairs:
                feature_names.append(f"{feat1}_x_{feat2}")
        
        return feature_names
    
    def get_feature_count(self) -> int:
        """
        Get total number of features after preprocessing.
        
        Returns:
            Feature count
        """
        return len(self.get_feature_names())
    
    def save(self, path: str):
        """
        Save preprocessor to disk.
        
        Args:
            path: Path to save preprocessor
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'numerical_imputer': self.numerical_imputer,
            'categorical_imputer': self.categorical_imputer,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_stats': self.feature_stats,
            'is_fitted': self.is_fitted,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: str, config) -> 'ClaimsPreprocessor':
        """
        Load preprocessor from disk.
        
        Args:
            path: Path to load from
            config: Configuration object
        
        Returns:
            Loaded preprocessor
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(config)
        preprocessor.numerical_features = state['numerical_features']
        preprocessor.categorical_features = state['categorical_features']
        preprocessor.numerical_imputer = state['numerical_imputer']
        preprocessor.categorical_imputer = state['categorical_imputer']
        preprocessor.scaler = state['scaler']
        preprocessor.label_encoders = state['label_encoders']
        preprocessor.feature_stats = state['feature_stats']
        preprocessor.is_fitted = state['is_fitted']
        
        logger.info(f"Preprocessor loaded from {path}")
        
        return preprocessor


class DataLoader:
    """
    PyTorch DataLoader wrapper for preprocessed data.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ):
        """
        Initialize data loader.
        
        Args:
            X: Input data array
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.X = X
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        """Iterate over batches"""
        indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield self.X[batch_indices]
    
    def __len__(self):
        """Number of batches"""
        return self.n_batches


if __name__ == "__main__":
    # Example usage
    from src.config_manager import ConfigManager
    from src.data_ingestion import load_sample_data
    
    # Load config
    config_manager = ConfigManager("config/example_config.yaml")
    config = config_manager.get_config()
    
    # Generate sample data
    df = load_sample_data(n_samples=1000)
    
    # Create and fit preprocessor
    preprocessor = ClaimsPreprocessor(config)
    X_train = preprocessor.fit_transform(df)
    
    print(f"Transformed data shape: {X_train.shape}")
    print(f"Feature names: {preprocessor.get_feature_names()}")
    
    # Save preprocessor
    preprocessor.save("models/preprocessor.pkl")
