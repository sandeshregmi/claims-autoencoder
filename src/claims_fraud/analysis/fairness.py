"""
Fairness Analysis Module
Detects bias and ensures fairness across protected groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats


logger = logging.getLogger(__name__)


class FairnessAnalyzer:
    """
    Analyzes model fairness across protected groups.
    
    Key Metrics:
    - Demographic Parity: P(fraud_flag=1 | group_A) ≈ P(fraud_flag=1 | group_B)
    - Equal Opportunity: P(fraud_flag=1 | actual_fraud=1, group_A) ≈ P(fraud_flag=1 | actual_fraud=1, group_B)
    - Predictive Parity: P(actual_fraud=1 | fraud_flag=1, group_A) ≈ P(actual_fraud=1 | fraud_flag=1, group_B)
    - Disparate Impact Ratio: Should be close to 1.0 (between 0.8 and 1.2 is acceptable)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        fraud_scores: np.ndarray,
        protected_attributes: List[str],
        threshold_percentile: float = 95.0
    ):
        """
        Initialize fairness analyzer.
        
        Args:
            data: DataFrame with protected attributes
            fraud_scores: Model fraud scores
            protected_attributes: List of protected attribute column names
            threshold_percentile: Percentile threshold for flagging fraud
        """
        self.data = data.copy()
        self.fraud_scores = fraud_scores
        self.protected_attributes = protected_attributes
        self.threshold_percentile = threshold_percentile
        
        # Calculate threshold
        self.threshold = np.percentile(fraud_scores, threshold_percentile)
        
        # Create binary fraud flags
        self.fraud_flags = (fraud_scores > self.threshold).astype(int)
        
        # Add to data
        self.data['fraud_score'] = fraud_scores
        self.data['fraud_flag'] = self.fraud_flags
    
    def analyze_attribute(
        self,
        attribute: str
    ) -> Dict:
        """
        Analyze fairness for a single protected attribute.
        
        Args:
            attribute: Name of protected attribute
            
        Returns:
            Dictionary with fairness metrics
        """
        if attribute not in self.data.columns:
            raise ValueError(f"Attribute {attribute} not found in data")
        
        # Get unique groups
        groups = self.data[attribute].dropna().unique()
        
        results = {
            'attribute': attribute,
            'groups': {},
            'pairwise_comparisons': [],
            'overall_metrics': {}
        }
        
        # Calculate metrics for each group
        for group in groups:
            group_mask = self.data[attribute] == group
            group_data = self.data[group_mask]
            
            n_total = len(group_data)
            n_flagged = group_data['fraud_flag'].sum()
            flag_rate = n_flagged / n_total if n_total > 0 else 0
            
            avg_score = group_data['fraud_score'].mean()
            median_score = group_data['fraud_score'].median()
            
            results['groups'][str(group)] = {
                'count': int(n_total),
                'flagged': int(n_flagged),
                'flag_rate': float(flag_rate),
                'avg_fraud_score': float(avg_score),
                'median_fraud_score': float(median_score),
                'percentile_95': float(np.percentile(group_data['fraud_score'], 95)),
                'percentile_99': float(np.percentile(group_data['fraud_score'], 99))
            }
        
        # Pairwise comparisons (Disparate Impact Ratio)
        group_list = list(groups)
        for i, group_a in enumerate(group_list):
            for group_b in group_list[i+1:]:
                rate_a = results['groups'][str(group_a)]['flag_rate']
                rate_b = results['groups'][str(group_b)]['flag_rate']
                
                # Disparate Impact Ratio
                # Ratio should be close to 1.0
                # Between 0.8 and 1.2 is generally acceptable
                if rate_b > 0:
                    di_ratio = rate_a / rate_b
                else:
                    di_ratio = np.inf
                
                # Statistical test (Chi-square)
                group_a_mask = self.data[attribute] == group_a
                group_b_mask = self.data[attribute] == group_b
                
                contingency_table = pd.crosstab(
                    self.data.loc[group_a_mask | group_b_mask, attribute],
                    self.data.loc[group_a_mask | group_b_mask, 'fraud_flag']
                )
                
                chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                
                # Effect size (Cohen's h for proportions)
                h = 2 * (np.arcsin(np.sqrt(rate_a)) - np.arcsin(np.sqrt(rate_b)))
                
                comparison = {
                    'group_a': str(group_a),
                    'group_b': str(group_b),
                    'rate_a': float(rate_a),
                    'rate_b': float(rate_b),
                    'disparate_impact_ratio': float(di_ratio),
                    'chi_square': float(chi2),
                    'p_value': float(p_value),
                    'effect_size_h': float(h),
                    'is_fair': bool(0.8 <= di_ratio <= 1.25 and p_value > 0.05)
                }
                
                results['pairwise_comparisons'].append(comparison)
        
        # Overall fairness assessment
        all_ratios = [c['disparate_impact_ratio'] for c in results['pairwise_comparisons'] 
                     if not np.isinf(c['disparate_impact_ratio'])]
        
        if all_ratios:
            min_ratio = min(all_ratios)
            max_ratio = max(all_ratios)
            
            # Overall fairness: all ratios should be between 0.8 and 1.25
            is_fair = all(0.8 <= r <= 1.25 for r in all_ratios)
            
            results['overall_metrics'] = {
                'min_disparate_impact_ratio': float(min_ratio),
                'max_disparate_impact_ratio': float(max_ratio),
                'is_fair': bool(is_fair),
                'fairness_score': float(1.0 - abs(1.0 - min_ratio))  # Closer to 1.0 is better
            }
        
        return results
    
    def analyze_all_attributes(self) -> Dict[str, Dict]:
        """
        Analyze fairness for all protected attributes.
        
        Returns:
            Dictionary mapping attribute names to fairness results
        """
        all_results = {}
        
        for attribute in self.protected_attributes:
            try:
                results = self.analyze_attribute(attribute)
                all_results[attribute] = results
            except Exception as e:
                logger.error(f"Error analyzing attribute {attribute}: {e}")
                all_results[attribute] = {'error': str(e)}
        
        return all_results
    
    def get_bias_summary(self) -> pd.DataFrame:
        """
        Get a summary of bias across all attributes.
        
        Returns:
            DataFrame with bias summary
        """
        all_results = self.analyze_all_attributes()
        
        summary_data = []
        
        for attribute, results in all_results.items():
            if 'error' in results:
                continue
            
            if 'overall_metrics' in results and results['overall_metrics']:
                metrics = results['overall_metrics']
                
                summary_data.append({
                    'attribute': attribute,
                    'num_groups': len(results['groups']),
                    'min_di_ratio': metrics.get('min_disparate_impact_ratio', 0),
                    'max_di_ratio': metrics.get('max_disparate_impact_ratio', 0),
                    'is_fair': metrics.get('is_fair', False),
                    'fairness_score': metrics.get('fairness_score', 0)
                })
        
        if summary_data:
            return pd.DataFrame(summary_data).sort_values('fairness_score', ascending=False)
        else:
            return pd.DataFrame()
    
    def get_detailed_comparison(
        self,
        attribute: str
    ) -> pd.DataFrame:
        """
        Get detailed comparison table for an attribute.
        
        Args:
            attribute: Protected attribute name
            
        Returns:
            DataFrame with detailed group comparisons
        """
        results = self.analyze_attribute(attribute)
        
        if 'groups' not in results:
            return pd.DataFrame()
        
        group_data = []
        for group, metrics in results['groups'].items():
            group_data.append({
                'group': group,
                'count': metrics['count'],
                'flagged': metrics['flagged'],
                'flag_rate': metrics['flag_rate'],
                'avg_score': metrics['avg_fraud_score'],
                'median_score': metrics['median_fraud_score'],
                'p95_score': metrics['percentile_95'],
                'p99_score': metrics['percentile_99']
            })
        
        return pd.DataFrame(group_data)
    
    def test_individual_fairness(
        self,
        sample_indices: Optional[List[int]] = None,
        n_samples: int = 100
    ) -> Dict:
        """
        Test individual fairness: similar individuals should get similar predictions.
        
        Args:
            sample_indices: Specific indices to test, or None for random
            n_samples: Number of samples to test if sample_indices is None
            
        Returns:
            Dictionary with individual fairness metrics
        """
        if sample_indices is None:
            sample_indices = np.random.choice(len(self.data), min(n_samples, len(self.data)), replace=False)
        
        # For each sample, find similar samples and compare scores
        # This is a simplified version - in practice, you'd define similarity more carefully
        
        consistency_scores = []
        
        for idx in sample_indices:
            sample_score = self.fraud_scores[idx]
            
            # Find similar samples (simplified: same protected group memberships)
            similar_mask = np.ones(len(self.data), dtype=bool)
            
            for attr in self.protected_attributes:
                if attr in self.data.columns:
                    similar_mask &= (self.data[attr] == self.data.iloc[idx][attr])
            
            similar_scores = self.fraud_scores[similar_mask]
            
            if len(similar_scores) > 1:
                # Calculate consistency (low std = high consistency)
                consistency = 1.0 / (1.0 + np.std(similar_scores))
                consistency_scores.append(consistency)
        
        return {
            'n_tested': len(sample_indices),
            'avg_consistency': float(np.mean(consistency_scores)) if consistency_scores else 0.0,
            'min_consistency': float(np.min(consistency_scores)) if consistency_scores else 0.0,
            'max_consistency': float(np.max(consistency_scores)) if consistency_scores else 0.0
        }
    
    def generate_fairness_report(
        self,
        attribute: str
    ) -> str:
        """
        Generate a human-readable fairness report.
        
        Args:
            attribute: Protected attribute to analyze
            
        Returns:
            Formatted report string
        """
        results = self.analyze_attribute(attribute)
        
        report = []
        report.append(f"FAIRNESS ANALYSIS REPORT: {attribute.upper()}")
        report.append("=" * 60)
        report.append("")
        
        # Group statistics
        report.append("GROUP STATISTICS:")
        report.append("-" * 60)
        for group, metrics in results['groups'].items():
            report.append(f"\n{group}:")
            report.append(f"  Total claims: {metrics['count']:,}")
            report.append(f"  Flagged as fraud: {metrics['flagged']:,} ({metrics['flag_rate']*100:.2f}%)")
            report.append(f"  Average fraud score: {metrics['avg_fraud_score']:,.2f}")
        
        report.append("\n")
        report.append("PAIRWISE COMPARISONS:")
        report.append("-" * 60)
        
        for comp in results['pairwise_comparisons']:
            report.append(f"\n{comp['group_a']} vs {comp['group_b']}:")
            report.append(f"  Flag rate: {comp['rate_a']*100:.2f}% vs {comp['rate_b']*100:.2f}%")
            report.append(f"  Disparate Impact Ratio: {comp['disparate_impact_ratio']:.3f}")
            report.append(f"  Statistical significance: p={comp['p_value']:.4f}")
            
            if comp['is_fair']:
                report.append(f"  ✅ FAIR (ratio between 0.8 and 1.25, p > 0.05)")
            else:
                report.append(f"  ⚠️ POTENTIAL BIAS DETECTED")
        
        report.append("\n")
        report.append("OVERALL ASSESSMENT:")
        report.append("-" * 60)
        
        if 'overall_metrics' in results and results['overall_metrics']:
            metrics = results['overall_metrics']
            report.append(f"Fairness Score: {metrics['fairness_score']:.3f}")
            
            if metrics['is_fair']:
                report.append("✅ Model appears fair across groups")
            else:
                report.append("⚠️ Potential bias detected - review recommended")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    sample_data = pd.DataFrame({
        'patient_gender': np.random.choice(['M', 'F'], n_samples),
        'patient_age_group': np.random.choice(['18-35', '36-50', '51-65', '65+'], n_samples),
        'geographic_region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    })
    
    # Generate fraud scores with some bias
    fraud_scores = np.random.lognormal(10, 2, n_samples)
    # Add bias: make females slightly more likely to be flagged
    fraud_scores[sample_data['patient_gender'] == 'F'] *= 1.2
    
    # Analyze fairness
    analyzer = FairnessAnalyzer(
        data=sample_data,
        fraud_scores=fraud_scores,
        protected_attributes=['patient_gender', 'patient_age_group', 'geographic_region'],
        threshold_percentile=95.0
    )
    
    # Get results
    print(analyzer.generate_fairness_report('patient_gender'))
    print("\n\n")
    
    bias_summary = analyzer.get_bias_summary()
    print("BIAS SUMMARY:")
    print(bias_summary)
