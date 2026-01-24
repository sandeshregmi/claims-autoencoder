"""
Claims Fraud Detection with Tree Models - Configurable Runner

Can run in two modes:
1. With sample data (demo mode): python tree_fraud_detection_runner.py
2. With your config file: python tree_fraud_detection_runner.py --config config/example_config.yaml

Author: ML Engineering Team
Date: 2026-01-22
"""

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.tree_models import ClaimsTreeAutoencoder, create_ensemble_fraud_scores
from src.config_manager import ConfigManager
from src.data_ingestion import DataIngestion
from src.preprocessing import ClaimsPreprocessor


def create_sample_data(n_samples=1000, n_fraud=50):
    """Create sample insurance claims data."""
    np.random.seed(42)
    
    n_normal = n_samples - n_fraud
    
    # Normal claims
    normal_claims = {
        'claim_amount': np.random.uniform(1000, 10000, n_normal),
        'patient_age': np.random.randint(25, 70, n_normal),
        'provider_experience_years': np.random.randint(5, 30, n_normal),
        'days_since_last_claim': np.random.randint(30, 365, n_normal),
        'num_previous_claims': np.random.randint(0, 10, n_normal),
        'average_claim_amount': np.random.uniform(1000, 8000, n_normal),
        'claim_duration_days': np.random.randint(1, 30, n_normal),
        'claim_type': np.random.choice(['routine', 'emergency', 'preventive'], n_normal),
        'provider_specialty': np.random.choice(['general', 'specialist', 'surgery'], n_normal),
        'diagnosis_code': np.random.choice(['D001', 'D002', 'D003'], n_normal),
        'procedure_code': np.random.choice(['P001', 'P002', 'P003'], n_normal),
        'patient_gender': np.random.choice(['M', 'F'], n_normal),
        'geographic_region': np.random.choice(['Northeast', 'South', 'Midwest', 'West'], n_normal),
    }
    
    normal_df = pd.DataFrame(normal_claims)
    normal_df['is_fraud'] = 0
    
    # Fraudulent claims
    fraud_claims = {
        'claim_amount': np.random.uniform(50000, 100000, n_fraud),
        'patient_age': np.random.randint(18, 90, n_fraud),
        'provider_experience_years': np.random.randint(1, 5, n_fraud),
        'days_since_last_claim': np.random.randint(1, 10, n_fraud),
        'num_previous_claims': np.random.randint(20, 50, n_fraud),
        'average_claim_amount': np.random.uniform(40000, 90000, n_fraud),
        'claim_duration_days': np.random.randint(1, 5, n_fraud),
        'claim_type': np.random.choice(['complex', 'experimental'], n_fraud),
        'provider_specialty': np.random.choice(['experimental', 'rare'], n_fraud),
        'diagnosis_code': np.random.choice(['D999', 'D998'], n_fraud),
        'procedure_code': np.random.choice(['P999', 'P998'], n_fraud),
        'patient_gender': np.random.choice(['M', 'F'], n_fraud),
        'geographic_region': np.random.choice(['Remote', 'International'], n_fraud),
    }
    
    fraud_df = pd.DataFrame(fraud_claims)
    fraud_df['is_fraud'] = 1
    
    # Combine and shuffle
    claims = pd.concat([normal_df, fraud_df], ignore_index=True)
    claims = claims.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return claims


def load_data_from_config(config_path):
    """Load data using configuration file."""
    print(f"\nLoading data from config: {config_path}")
    
    # Load config
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    # Load data
    data_ingestion = DataIngestion(config)
    train_df, val_df, test_df = data_ingestion.load_train_val_test()
    
    # Combine for fraud detection (or use just training data)
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print(f"  Loaded {len(df)} claims")
    print(f"  Features: {list(df.columns)}")
    
    # Get feature types from config
    categorical_features = config.data.categorical_features
    numerical_features = config.data.numerical_features
    
    # Check if there's a fraud label
    has_labels = 'is_fraud' in df.columns or 'fraud' in df.columns or 'label' in df.columns
    
    return df, categorical_features, numerical_features, has_labels, config


def run_fraud_detection(df, categorical_features, numerical_features, model_type='xgboost', has_labels=False):
    """Run fraud detection on claims data."""
    
    print("\n" + "="*80)
    print(f"FRAUD DETECTION WITH {model_type.upper()}")
    print("="*80)
    
    # Separate features and labels if available
    if has_labels:
        label_col = 'is_fraud' if 'is_fraud' in df.columns else ('fraud' if 'fraud' in df.columns else 'label')
        X = df.drop(label_col, axis=1)
        y_true = df[label_col].values
    else:
        X = df
        y_true = None
    
    # Train model
    print(f"\nTraining {model_type} fraud detector...")
    model = ClaimsTreeAutoencoder(
        model_type=model_type,
        n_estimators=100 if model_type == 'xgboost' else None,
        iterations=100 if model_type == 'catboost' else None
    )
    
    model.fit(
        X,
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )
    
    # Compute fraud scores
    print("\nComputing fraud scores...")
    fraud_scores, per_feature_errors = model.compute_fraud_scores(X)
    
    # Statistics
    print(f"\nFraud Score Statistics:")
    print(f"  Mean:   {fraud_scores.mean():.4f}")
    print(f"  Median: {np.median(fraud_scores):.4f}")
    print(f"  Std:    {fraud_scores.std():.4f}")
    print(f"  Min:    {fraud_scores.min():.4f}")
    print(f"  Max:    {fraud_scores.max():.4f}")
    
    # Percentiles
    print(f"\nFraud Score Percentiles:")
    for p in [90, 95, 99]:
        threshold = np.percentile(fraud_scores, p)
        n_flagged = (fraud_scores > threshold).sum()
        print(f"  {p}th percentile: {threshold:.4f} ({n_flagged} claims flagged)")
    
    # If we have labels, evaluate performance
    if y_true is not None:
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        threshold = np.percentile(fraud_scores, 95)
        predictions = (fraud_scores > threshold).astype(int)
        
        print(f"\nPerformance Metrics (threshold at 95th percentile):")
        print(f"  AUC-ROC:   {roc_auc_score(y_true, fraud_scores):.4f}")
        print(f"  Precision: {precision_score(y_true, predictions):.4f}")
        print(f"  Recall:    {recall_score(y_true, predictions):.4f}")
        print(f"  F1 Score:  {f1_score(y_true, predictions):.4f}")
    
    # Show top suspicious claims
    print(f"\n🚨 Top 10 Most Suspicious Claims:")
    top_indices = np.argsort(fraud_scores)[-10:][::-1]
    display_cols = ['claim_amount', 'claim_type', 'num_previous_claims']
    if has_labels:
        display_cols.append(label_col)
    
    # Only show columns that exist
    display_cols = [col for col in display_cols if col in df.columns]
    top_claims = df.iloc[top_indices][display_cols]
    top_claims['fraud_score'] = fraud_scores[top_indices]
    print(top_claims.to_string())
    
    # Feature importance
    print(f"\n🔍 Top Fraud Indicators:")
    for feature in X.columns[:3]:  # Show top 3 features
        top_indicators = model.get_top_fraud_indicators(feature, top_k=3)
        print(f"\n  For predicting {feature}:")
        for i, (indicator, importance) in enumerate(top_indicators, 1):
            print(f"    {i}. {indicator}: {importance:.4f}")
    
    return model, fraud_scores


def run_ensemble(df, categorical_features, numerical_features, has_labels=False):
    """Run ensemble fraud detection."""
    
    print("\n" + "="*80)
    print("ENSEMBLE FRAUD DETECTION (XGBoost + CatBoost)")
    print("="*80)
    
    # Separate features and labels if available
    if has_labels:
        label_col = 'is_fraud' if 'is_fraud' in df.columns else ('fraud' if 'fraud' in df.columns else 'label')
        X = df.drop(label_col, axis=1)
        y_true = df[label_col].values
    else:
        X = df
        y_true = None
    
    # Train both models
    models = {}
    
    print("\nTraining ensemble models...")
    for model_type in ['xgboost', 'catboost']:
        print(f"  Training {model_type}...")
        model = ClaimsTreeAutoencoder(
            model_type=model_type,
            n_estimators=100 if model_type == 'xgboost' else None,
            iterations=100 if model_type == 'catboost' else None
        )
        model.fit(X, categorical_features=categorical_features,
                 numerical_features=numerical_features, verbose=False)
        models[model_type] = model
    
    # Individual performance
    print("\nIndividual Model Performance:")
    print("-" * 80)
    
    if y_true is not None:
        from sklearn.metrics import roc_auc_score
        
        for name, model in models.items():
            scores, _ = model.compute_fraud_scores(X)
            auc = roc_auc_score(y_true, scores)
            print(f"  {name.upper():12s} AUC-ROC: {auc:.4f}")
    
    # Ensemble predictions
    print("\nEnsemble Performance:")
    print("-" * 80)
    
    weight_configs = [
        {'xgboost': 0.5, 'catboost': 0.5},
        {'xgboost': 0.7, 'catboost': 0.3},
    ]
    
    for weights in weight_configs:
        ensemble_scores = create_ensemble_fraud_scores(models, X, weights=weights)
        weight_str = ", ".join([f"{k}={v}" for k, v in weights.items()])
        
        if y_true is not None:
            auc = roc_auc_score(y_true, ensemble_scores)
            print(f"  Weights ({weight_str}): AUC-ROC = {auc:.4f}")
        else:
            print(f"  Weights ({weight_str}): Mean score = {ensemble_scores.mean():.4f}")
    
    return models


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run fraud detection with tree models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample data (demo mode)
  python tree_fraud_detection_runner.py
  
  # Run with your config file
  python tree_fraud_detection_runner.py --config config/example_config.yaml
  
  # Run specific model
  python tree_fraud_detection_runner.py --config config/example_config.yaml --model xgboost
  
  # Run ensemble
  python tree_fraud_detection_runner.py --config config/example_config.yaml --ensemble
        """
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (optional, uses sample data if not provided)')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'catboost'],
                       help='Model to use (default: xgboost)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Run ensemble detection (XGBoost + CatBoost)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CLAIMS FRAUD DETECTION WITH TREE MODELS")
    print("="*80)
    
    # Load data
    if args.config:
        # Use config file
        df, categorical_features, numerical_features, has_labels, config = load_data_from_config(args.config)
    else:
        # Use sample data
        print("\nNo config provided - using sample data (demo mode)")
        print("  To use your data: --config config/example_config.yaml")
        
        df = create_sample_data(n_samples=1000, n_fraud=50)
        categorical_features = [
            'claim_type', 'provider_specialty', 'diagnosis_code',
            'procedure_code', 'patient_gender', 'geographic_region'
        ]
        numerical_features = [
            'claim_amount', 'patient_age', 'provider_experience_years',
            'days_since_last_claim', 'num_previous_claims',
            'average_claim_amount', 'claim_duration_days'
        ]
        has_labels = True
        
        print(f"\nGenerated {len(df)} sample claims ({df['is_fraud'].sum()} fraudulent)")
    
    # Run fraud detection
    if args.ensemble:
        run_ensemble(df, categorical_features, numerical_features, has_labels)
    else:
        run_fraud_detection(df, categorical_features, numerical_features, args.model, has_labels)
    
    print("\n" + "="*80)
    print("FRAUD DETECTION COMPLETE")
    print("="*80)
    print("\n💡 Next Steps:")
    print("  • Review suspicious claims above")
    print("  • Adjust threshold based on business needs")
    print("  • Investigate high-scoring claims")
    print("  • Save model for production: model.save('fraud_detector')")
    
    if not args.config:
        print("\n  To run on your data:")
        print("    python tree_fraud_detection_runner.py --config config/example_config.yaml")


if __name__ == "__main__":
    main()
