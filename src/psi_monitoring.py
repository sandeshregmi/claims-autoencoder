"""
PSI Monitoring Module
Population Stability Index for drift detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


logger = logging.getLogger(__name__)


class PSIMonitor:
    """
    Population Stability Index (PSI) calculator for drift detection.
    
    PSI measures the shift in distribution between two datasets.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Minor change
    PSI >= 0.2: Major change (consider retraining)
    """
    
    def __init__(
        self,
        reference_data: np.ndarray,
        num_bins: int = 10,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize PSI monitor with reference data.
        
        Args:
            reference_data: Reference dataset (training data)
            num_bins: Number of bins for discretization
            feature_names: Names of features
        """
        self.reference_data = reference_data
        self.num_bins = num_bins
        self.n_features = reference_data.shape[1]
        
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        else:
            self.feature_names = feature_names
        
        # Compute reference distributions
        self.reference_distributions = self._compute_distributions(reference_data)
    
    def _compute_distributions(
        self,
        data: np.ndarray
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute histogram distributions for each feature.
        
        Args:
            data: Input data
        
        Returns:
            Dictionary mapping feature index to (bin_edges, counts)
        """
        distributions = {}
        
        for i in range(self.n_features):
            feature_data = data[:, i]
            
            # Remove NaN values
            feature_data = feature_data[~np.isnan(feature_data)]
            
            if len(feature_data) == 0:
                logger.warning(f"Feature {i} has no valid values")
                continue
            
            # Compute histogram
            counts, bin_edges = np.histogram(feature_data, bins=self.num_bins)
            
            # Convert to proportions
            proportions = counts / counts.sum()
            
            distributions[i] = (bin_edges, proportions)
        
        return distributions
    
    def calculate_psi(
        self,
        current_data: np.ndarray,
        epsilon: float = 1e-10
    ) -> Dict[str, float]:
        """
        Calculate PSI for each feature.
        
        Args:
            current_data: Current dataset to compare
            epsilon: Small value to avoid division by zero
        
        Returns:
            Dictionary mapping feature names to PSI values
        """
        if current_data.shape[1] != self.n_features:
            raise ValueError(
                f"Current data has {current_data.shape[1]} features, "
                f"expected {self.n_features}"
            )
        
        psi_values = {}
        
        for i in range(self.n_features):
            if i not in self.reference_distributions:
                continue
            
            bin_edges, ref_proportions = self.reference_distributions[i]
            
            # Get current feature data
            feature_data = current_data[:, i]
            feature_data = feature_data[~np.isnan(feature_data)]
            
            if len(feature_data) == 0:
                logger.warning(f"Feature {i} has no valid values in current data")
                continue
            
            # Compute current distribution using same bins
            curr_counts, _ = np.histogram(feature_data, bins=bin_edges)
            curr_proportions = curr_counts / curr_counts.sum()
            
            # Calculate PSI
            # PSI = sum((actual% - expected%) * ln(actual% / expected%))
            psi = 0.0
            for ref_prop, curr_prop in zip(ref_proportions, curr_proportions):
                # Add epsilon to avoid log(0)
                ref_prop = max(ref_prop, epsilon)
                curr_prop = max(curr_prop, epsilon)
                
                psi += (curr_prop - ref_prop) * np.log(curr_prop / ref_prop)
            
            psi_values[self.feature_names[i]] = float(psi)
        
        return psi_values
    
    def calculate_overall_psi(self, current_data: np.ndarray) -> float:
        """
        Calculate overall PSI across all features.
        
        Args:
            current_data: Current dataset
        
        Returns:
            Overall PSI (average across features)
        """
        psi_values = self.calculate_psi(current_data)
        
        if not psi_values:
            return 0.0
        
        return float(np.mean(list(psi_values.values())))
    
    def detect_drift(
        self,
        current_data: np.ndarray,
        threshold_minor: float = 0.1,
        threshold_major: float = 0.2
    ) -> Dict[str, Dict]:
        """
        Detect drift in features.
        
        Args:
            current_data: Current dataset
            threshold_minor: Threshold for minor drift
            threshold_major: Threshold for major drift
        
        Returns:
            Dictionary with drift detection results
        """
        psi_values = self.calculate_psi(current_data)
        
        results = {
            'psi_values': psi_values,
            'overall_psi': self.calculate_overall_psi(current_data),
            'drifted_features': {
                'minor': [],
                'major': []
            }
        }
        
        for feature_name, psi in psi_values.items():
            if psi >= threshold_major:
                results['drifted_features']['major'].append(feature_name)
            elif psi >= threshold_minor:
                results['drifted_features']['minor'].append(feature_name)
        
        # Overall drift status
        if results['overall_psi'] >= threshold_major:
            results['drift_status'] = 'major'
        elif results['overall_psi'] >= threshold_minor:
            results['drift_status'] = 'minor'
        else:
            results['drift_status'] = 'stable'
        
        return results
    
    def plot_psi_scores(
        self,
        psi_values: Dict[str, float],
        threshold_minor: float = 0.1,
        threshold_major: float = 0.2,
        save_path: Optional[str] = None
    ):
        """
        Plot PSI scores for all features.
        
        Args:
            psi_values: Dictionary of PSI values
            threshold_minor: Minor drift threshold
            threshold_major: Major drift threshold
            save_path: Path to save plot
        """
        features = list(psi_values.keys())
        psi_scores = list(psi_values.values())
        
        # Create color map based on thresholds
        colors = []
        for psi in psi_scores:
            if psi >= threshold_major:
                colors.append('red')
            elif psi >= threshold_minor:
                colors.append('orange')
            else:
                colors.append('green')
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(features)), psi_scores, color=colors, alpha=0.7)
        
        # Add threshold lines
        plt.axhline(y=threshold_minor, color='orange', linestyle='--',
                   label=f'Minor Drift ({threshold_minor})')
        plt.axhline(y=threshold_major, color='red', linestyle='--',
                   label=f'Major Drift ({threshold_major})')
        
        plt.xlabel('Feature')
        plt.ylabel('PSI Score')
        plt.title('Population Stability Index by Feature')
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_distribution_comparison(
        self,
        current_data: np.ndarray,
        feature_idx: int,
        save_path: Optional[str] = None
    ):
        """
        Plot distribution comparison for a specific feature.
        
        Args:
            current_data: Current dataset
            feature_idx: Index of feature to plot
            save_path: Path to save plot
        """
        if feature_idx not in self.reference_distributions:
            raise ValueError(f"Feature {feature_idx} not in reference distributions")
        
        bin_edges, ref_proportions = self.reference_distributions[feature_idx]
        
        # Get current distribution
        feature_data = current_data[:, feature_idx]
        feature_data = feature_data[~np.isnan(feature_data)]
        curr_counts, _ = np.histogram(feature_data, bins=bin_edges)
        curr_proportions = curr_counts / curr_counts.sum()
        
        # Calculate PSI for this feature
        psi_values = self.calculate_psi(current_data)
        psi = psi_values.get(self.feature_names[feature_idx], 0.0)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(ref_proportions))
        width = 0.35
        
        plt.bar(x - width/2, ref_proportions, width, label='Reference',
               alpha=0.7, color='blue')
        plt.bar(x + width/2, curr_proportions, width, label='Current',
               alpha=0.7, color='orange')
        
        plt.xlabel('Bin')
        plt.ylabel('Proportion')
        plt.title(f'Distribution Comparison: {self.feature_names[feature_idx]}\n'
                 f'PSI = {psi:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_drift_report(
        self,
        current_data: np.ndarray,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive drift detection report.
        
        Args:
            current_data: Current dataset
            output_dir: Directory to save plots
        
        Returns:
            Dictionary with drift results
        """
        results = self.detect_drift(current_data)
        
        logger.info(f"Overall PSI: {results['overall_psi']:.4f}")
        logger.info(f"Drift status: {results['drift_status']}")
        logger.info(f"Features with minor drift: {len(results['drifted_features']['minor'])}")
        logger.info(f"Features with major drift: {len(results['drifted_features']['major'])}")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot PSI scores
            self.plot_psi_scores(
                results['psi_values'],
                save_path=str(output_dir / 'psi_scores.png')
            )
            
            # Plot distribution comparisons for drifted features
            for feature_name in results['drifted_features']['major']:
                feature_idx = self.feature_names.index(feature_name)
                self.plot_distribution_comparison(
                    current_data,
                    feature_idx,
                    save_path=str(output_dir / f'dist_{feature_name}.png')
                )
        
        return results


if __name__ == "__main__":
    # Example usage
    # Generate sample data
    np.random.seed(42)
    
    # Reference data (training)
    reference_data = np.random.randn(10000, 5)
    
    # Current data (slightly shifted)
    current_data = np.random.randn(5000, 5) + 0.3  # Added shift
    
    # Create monitor
    feature_names = [f"feature_{i}" for i in range(5)]
    monitor = PSIMonitor(reference_data, num_bins=10, feature_names=feature_names)
    
    # Detect drift
    results = monitor.generate_drift_report(current_data, output_dir="outputs/drift")
    
    print("\nDrift Detection Results:")
    print(f"Overall PSI: {results['overall_psi']:.4f}")
    print(f"Status: {results['drift_status']}")
    print(f"\nPSI by feature:")
    for feature, psi in results['psi_values'].items():
        print(f"  {feature}: {psi:.4f}")
