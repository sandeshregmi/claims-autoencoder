"""
Business Rules Engine
Applies business logic for fraud detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class BusinessRulesEngine:
    """
    Applies business rules for fraud detection.
    
    Features:
    - Threshold-based fraud detection
    - Frequency-based alerts
    - Pattern-based anomaly detection
    - Configurable alert triggers
    """
    
    def __init__(self, config):
        """
        Initialize rules engine with configuration.
        
        Args:
            config: Configuration object containing business rules
        """
        self.config = config
        self.business_rules = config.business_rules if hasattr(config, 'business_rules') else None
        
        if not self.business_rules:
            logger.warning("No business rules configured")
            self.fraud_thresholds = {}
            self.alert_triggers = {}
        else:
            self.fraud_thresholds = self.business_rules.get('fraud_thresholds', {})
            self.alert_triggers = self.business_rules.get('alert_triggers', {})
    
    def apply_all_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all business rules and add flags to DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with rule flags added
        """
        df = df.copy()
        
        # Apply each rule
        df = self._check_high_value_claims(df)
        df = self._check_rapid_claims(df)
        df = self._check_age_anomalies(df)
        df = self._check_provider_experience(df)
        
        # Create overall risk score
        df = self._calculate_risk_score(df)
        
        logger.info(f"Business rules applied to {len(df)} claims")
        
        return df
    
    def _check_high_value_claims(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag high-value claims."""
        if 'claim_amount' not in df.columns:
            return df
        
        high_risk_threshold = self.fraud_thresholds.get('claim_amount_high_risk', 100000)
        medium_risk_threshold = self.fraud_thresholds.get('claim_amount_medium_risk', 50000)
        
        df['flag_high_value'] = df['claim_amount'] >= high_risk_threshold
        df['flag_medium_value'] = (df['claim_amount'] >= medium_risk_threshold) & \
                                  (df['claim_amount'] < high_risk_threshold)
        
        high_count = df['flag_high_value'].sum()
        medium_count = df['flag_medium_value'].sum()
        
        if high_count > 0:
            logger.info(f"Flagged {high_count} high-value claims (>= ${high_risk_threshold:,})")
        if medium_count > 0:
            logger.info(f"Flagged {medium_count} medium-value claims (>= ${medium_risk_threshold:,})")
        
        return df
    
    def _check_rapid_claims(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag rapid claim submissions."""
        if 'num_previous_claims' not in df.columns:
            return df
        
        max_per_month = self.fraud_thresholds.get('max_claims_per_month', 5)
        max_per_week = self.fraud_thresholds.get('max_claims_per_week', 2)
        
        # Flag based on previous claims count (simplified)
        df['flag_rapid_claims'] = df['num_previous_claims'] > max_per_month
        
        rapid_count = df['flag_rapid_claims'].sum()
        if rapid_count > 0:
            logger.info(f"Flagged {rapid_count} claims with rapid submission pattern")
        
        return df
    
    def _check_age_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag age-related anomalies."""
        if 'patient_age' not in df.columns:
            return df
        
        min_age = self.fraud_thresholds.get('patient_age_min', 0)
        max_age = self.fraud_thresholds.get('patient_age_max', 120)
        suspicious_ranges = self.fraud_thresholds.get('suspicious_age_ranges', [])
        
        # Flag out-of-range ages
        df['flag_age_anomaly'] = (df['patient_age'] < min_age) | (df['patient_age'] > max_age)
        
        # Flag suspicious age ranges
        df['flag_suspicious_age'] = False
        for age_range in suspicious_ranges:
            if len(age_range) == 2:
                min_r, max_r = age_range
                df.loc[(df['patient_age'] >= min_r) & (df['patient_age'] <= max_r), 
                      'flag_suspicious_age'] = True
        
        anomaly_count = df['flag_age_anomaly'].sum()
        suspicious_count = df['flag_suspicious_age'].sum()
        
        if anomaly_count > 0:
            logger.warning(f"Flagged {anomaly_count} age anomalies (out of valid range)")
        if suspicious_count > 0:
            logger.info(f"Flagged {suspicious_count} suspicious age values")
        
        return df
    
    def _check_provider_experience(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag provider experience anomalies."""
        if 'provider_experience_years' not in df.columns:
            return df
        
        suspicious_low = self.fraud_thresholds.get('suspicious_low_experience', 1)
        
        df['flag_low_provider_experience'] = df['provider_experience_years'] < suspicious_low
        
        low_exp_count = df['flag_low_provider_experience'].sum()
        if low_exp_count > 0:
            logger.info(f"Flagged {low_exp_count} claims from low-experience providers")
        
        return df
    
    def _calculate_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall risk score based on all flags."""
        flag_columns = [col for col in df.columns if col.startswith('flag_')]
        
        if not flag_columns:
            df['business_rule_risk_score'] = 0
            return df
        
        # Simple risk score: count of triggered flags
        df['business_rule_risk_score'] = df[flag_columns].sum(axis=1)
        
        # Categorize risk level
        df['business_rule_risk_level'] = 'LOW'
        df.loc[df['business_rule_risk_score'] >= 2, 'business_rule_risk_level'] = 'MEDIUM'
        df.loc[df['business_rule_risk_score'] >= 4, 'business_rule_risk_level'] = 'HIGH'
        
        risk_summary = df['business_rule_risk_level'].value_counts()
        logger.info(f"Risk distribution: {risk_summary.to_dict()}")
        
        return df
    
    def get_flagged_claims(
        self, 
        df: pd.DataFrame, 
        min_flags: int = 1
    ) -> pd.DataFrame:
        """
        Get claims that triggered business rules.
        
        Args:
            df: DataFrame with business rule flags
            min_flags: Minimum number of flags to include
            
        Returns:
            Filtered DataFrame with flagged claims
        """
        if 'business_rule_risk_score' not in df.columns:
            df = self.apply_all_rules(df)
        
        flagged = df[df['business_rule_risk_score'] >= min_flags].copy()
        
        logger.info(f"Found {len(flagged)} claims with >={min_flags} flags")
        
        return flagged
    
    def generate_alerts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate alert notifications based on triggers.
        
        Args:
            df: DataFrame with business rule flags
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # High value claim alerts
        high_value_threshold = self.alert_triggers.get('high_value_claim', 100000)
        if 'claim_amount' in df.columns:
            high_value_claims = df[df['claim_amount'] >= high_value_threshold]
            for idx, claim in high_value_claims.iterrows():
                alerts.append({
                    'type': 'high_value',
                    'severity': 'critical',
                    'claim_index': idx,
                    'message': f"High-value claim detected: ${claim['claim_amount']:,.2f}",
                    'amount': float(claim['claim_amount'])
                })
        
        # Rapid claims alerts
        rapid_threshold = self.alert_triggers.get('rapid_claims_count', 10)
        if 'num_previous_claims' in df.columns:
            rapid_claims = df[df['num_previous_claims'] >= rapid_threshold]
            for idx, claim in rapid_claims.iterrows():
                alerts.append({
                    'type': 'rapid_claims',
                    'severity': 'warning',
                    'claim_index': idx,
                    'message': f"Rapid claim pattern: {claim['num_previous_claims']} previous claims",
                    'count': int(claim['num_previous_claims'])
                })
        
        logger.info(f"Generated {len(alerts)} alerts")
        
        return alerts
    
    def get_rule_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of applied business rules.
        
        Args:
            df: DataFrame with business rule flags
            
        Returns:
            Summary dictionary
        """
        flag_columns = [col for col in df.columns if col.startswith('flag_')]
        
        summary = {
            'total_claims': len(df),
            'flags_applied': {},
            'risk_distribution': {}
        }
        
        # Count flags
        for col in flag_columns:
            flag_name = col.replace('flag_', '')
            summary['flags_applied'][flag_name] = int(df[col].sum())
        
        # Risk distribution
        if 'business_rule_risk_level' in df.columns:
            summary['risk_distribution'] = df['business_rule_risk_level'].value_counts().to_dict()
        
        return summary


def apply_business_rules(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Convenience function to apply business rules.
    
    Args:
        df: Input DataFrame
        config: Configuration object
        
    Returns:
        DataFrame with business rule flags
    """
    engine = BusinessRulesEngine(config)
    df_flagged = engine.apply_all_rules(df)
    
    # Log summary
    summary = engine.get_rule_summary(df_flagged)
    logger.info(f"Business Rules Summary: {summary}")
    
    return df_flagged


if __name__ == "__main__":
    # Example usage
    from claims_fraud.config.manager import ConfigManager
    
    # Load config
    config_manager = ConfigManager("config/config.yaml")
    config = config_manager.get_config()
    
    # Create sample data
    data = {
        'claim_amount': [500, 75000, 150000, 2000, 120000],
        'patient_age': [25, 45, 0, 105, 50],
        'num_previous_claims': [0, 3, 15, 2, 8],
        'provider_experience_years': [10, 0, 20, 5, 15]
    }
    df = pd.DataFrame(data)
    
    # Apply rules
    engine = BusinessRulesEngine(config)
    df_flagged = engine.apply_all_rules(df)
    
    print("\nFlagged claims:")
    print(df_flagged[[col for col in df_flagged.columns if 'flag_' in col or 'risk' in col]])
    
    # Generate alerts
    alerts = engine.generate_alerts(df_flagged)
    print(f"\nAlerts generated: {len(alerts)}")
    for alert in alerts:
        print(f"  - {alert['severity'].upper()}: {alert['message']}")
