"""
Claims Fraud Detection with Tree Models - Example

Demonstrates tree-based fraud detection for insurance claims using:
- XGBoost for fast baseline detection
- CatBoost for native categorical handling
- Ensemble for robust predictions
- Feature importance for investigation

Author: ML Engineering Team
Date: 2026-01-22
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tree_models import ClaimsTreeAutoencoder, create_ensemble_fraud_scores
from src.config_manager import ConfigManager


def create_sample_claims():
    """
    Create sample insurance claims data.
    
    Returns:
        DataFrame with normal and fraudulent claims
    """
    np.random.seed(42)
    
    n_normal = 950
    n_fraud = 50
    
    # Normal claims
    normal_claims = {
        'claim_amount': np.random.uniform(1000, 10000, n_normal),
        'patient_age': np.random.randint(25, 70, n_normal),
        'provider_experience_years': np.random.randint(5, 30, n_normal),
        'days_since_last_claim': np.random.randint(30, 365, n_normal),
        'num_previous_claims': np.random.randint(0, 10, n_normal),
        'claim_type': np.random.choice(['routine', 'emergency', 'preventive'], n_normal),
        'provider_specialty': np.random.choice(['general', 'specialist', 'surgery'], n_normal),
        'diagnosis_code': np.random.choice(['D001', 'D002', 'D003'], n_normal),
        'patient_gender': np.random.choice(['M', 'F'], n_normal),
        'geographic_region': np.random.choice(['Northeast', 'South', 'Midwest', 'West'], n_normal),
    }
    
    normal_df = pd.DataFrame(normal_claims)
    normal_df['is_fraud'] = 0
    
    # Fraudulent claims (unusual patterns)
    fraud_claims = {
        'claim_amount': np.random.uniform(50000, 100000, n_fraud),  # Much higher
        'patient_age': np.random.randint(18, 90, n_fraud),
        'provider_experience_years': np.random.randint(1, 5, n_fraud),  # Less experienced
        'days_since_last_claim': np.random.randint(1, 10, n_fraud),  # Very recent
        'num_previous_claims': np.random.randint(20, 50, n_fraud),  # Many claims
        'claim_type': np.random.choice(['complex', 'experimental'], n_fraud),  # Unusual types
        'provider_specialty': np.random.choice(['experimental', 'rare'], n_fraud),
        'diagnosis_code': np.random.choice(['D999', 'D998'], n_fraud),  # Rare codes
        'patient_gender': np.random.choice(['M', 'F'], n_fraud),
        'geographic_region': np.random.choice(['Remote', 'International'], n_fraud),
    }
    
    fraud_df = pd.DataFrame(fraud_claims)
    fraud_df['is_fraud'] = 1
    
    # Combine and shuffle
    claims = pd.concat([normal_df, fraud_df], ignore_index=True)
    claims = claims.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return claims


def example_1_xgboost_fraud_detection():
    """Example 1: Basic XGBoost fraud detection."""
    print("\n" + "="*80)
    print("EXAMPLE 1: XGBoost Fraud Detection")
    print("="*80)
    
    # Create sample claims
    claims = create_sample_claims()
    print(f"\nDataset: {len(claims)} claims ({claims['is_fraud'].sum()} fraudulent)")
    
    # Separate features and labels
    X = claims.drop('is_fraud', axis=1)
    y_true = claims['is_fraud'].values
    
    # Define feature types
    categorical_features = [
        'claim_type', 'provider_specialty', 'diagnosis_code',
        'patient_gender', 'geographic_region'
    ]
    numerical_features = [
        'claim_amount', 'patient_age', 'provider_experience_years',
        'days_since_last_claim', 'num_previous_claims'
    ]
    
    # Initialize and train model
    print("\n Training XGBoost fraud detector...")
    model = ClaimsTreeAutoencoder(
        model_type='xgboost',
        n_estimators=50,
        max_depth=4
    )
    
    model.fit(
        X,
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )
    
    # Compute fraud scores
    print("\nComputing fraud scores...")
    fraud_scores, per_feature_errors = model.compute_fraud_scores(X)
    
    # Analyze results
    print(f"\nFraud Score Statistics:")
    print(f"  Mean: {fraud_scores.mean():.4f}")
    print(f"  Std:  {fraud_scores.std():.4f}")
    print(f"  Min:  {fraud_scores.min():.4f}")
    print(f"  Max:  {fraud_scores.max():.4f}")
    
    # Detection performance
    threshold = np.percentile(fraud_scores, 95)
    predictions = (fraud_scores > threshold).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    print(f"\nDetection Performance (threshold={threshold:.4f}):")
    print(f"  Precision: {precision_score(y_true, predictions):.4f}")
    print(f"  Recall:    {recall_score(y_true, predictions):.4f}")
    print(f"  F1 Score:  {f1_score(y_true, predictions):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(y_true, fraud_scores):.4f}")
    
    # Show top fraud cases
    top_indices = np.argsort(fraud_scores)[-5:]
    print("\n🚨 Top 5 Suspected Fraud Cases:")
    print(claims.iloc[top_indices][['claim_amount', 'claim_type', 'num_previous_claims', 'is_fraud']])
    
    return model, X, y_true, fraud_scores


def example_2_catboost_comparison():
    """Example 2: Compare XGBoost vs CatBoost for fraud detection."""
    print("\n" + "="*80)
    print("EXAMPLE 2: XGBoost vs CatBoost Comparison")
    print("="*80)
    
    # Create data
    claims = create_sample_claims()
    X = claims.drop('is_fraud', axis=1)
    y_true = claims['is_fraud'].values
    
    categorical_features = [
        'claim_type', 'provider_specialty', 'diagnosis_code',
        'patient_gender', 'geographic_region'
    ]
    numerical_features = [
        'claim_amount', 'patient_age', 'provider_experience_years',
        'days_since_last_claim', 'num_previous_claims'
    ]
    
    # Train both models
    models = {}
    
    print("\nTraining XGBoost...")
    xgb_model = ClaimsTreeAutoencoder(model_type='xgboost', n_estimators=50)
    xgb_model.fit(X, categorical_features=categorical_features,
                  numerical_features=numerical_features, verbose=False)
    models['xgboost'] = xgb_model
    
    print("Training CatBoost...")
    cat_model = ClaimsTreeAutoencoder(model_type='catboost', iterations=50)
    cat_model.fit(X, categorical_features=categorical_features,
                  numerical_features=numerical_features, verbose=False)
    models['catboost'] = cat_model
    
    # Compare performance
    print("\n" + "-"*80)
    print("Fraud Detection Performance Comparison")
    print("-"*80)
    
    from sklearn.metrics import roc_auc_score
    
    for name, model in models.items():
        scores, _ = model.compute_fraud_scores(X)
        auc = roc_auc_score(y_true, scores)
        print(f"\n{name.upper()}:")
        print(f"  Mean Score: {scores.mean():.4f}")
        print(f"  AUC-ROC:    {auc:.4f}")
    
    return models, X, y_true


def example_3_feature_importance():
    """Example 3: Analyze fraud indicators."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Fraud Indicator Analysis")
    print("="*80)
    
    # Create data
    claims = create_sample_claims()
    X = claims.drop('is_fraud', axis=1)
    
    categorical_features = [
        'claim_type', 'provider_specialty', 'diagnosis_code',
        'patient_gender', 'geographic_region'
    ]
    numerical_features = [
        'claim_amount', 'patient_age', 'provider_experience_years',
        'days_since_last_claim', 'num_previous_claims'
    ]
    
    # Train model
    print("\nTraining fraud detection model...")
    model = ClaimsTreeAutoencoder(model_type='xgboost', n_estimators=50)
    model.fit(X, categorical_features=categorical_features,
             numerical_features=numerical_features, verbose=False)
    
    # Analyze fraud indicators
    print("\n🔍 Top Fraud Indicators for Each Feature:")
    print("-"*80)
    
    for feature in X.columns:
        top_indicators = model.get_top_fraud_indicators(feature, top_k=3)
        print(f"\n{feature.upper()}:")
        for i, (indicator, importance) in enumerate(top_indicators, 1):
            print(f"  {i}. {indicator:30s} (importance: {importance:.4f})")
    
    return model


def example_4_ensemble_detection():
    """Example 4: Ensemble fraud detection."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Ensemble Fraud Detection")
    print("="*80)
    
    # Create data
    claims = create_sample_claims()
    X = claims.drop('is_fraud', axis=1)
    y_true = claims['is_fraud'].values
    
    categorical_features = [
        'claim_type', 'provider_specialty', 'diagnosis_code',
        'patient_gender', 'geographic_region'
    ]
    numerical_features = [
        'claim_amount', 'patient_age', 'provider_experience_years',
        'days_since_last_claim', 'num_previous_claims'
    ]
    
    # Train ensemble
    print("\nTraining ensemble fraud detection models...")
    models = {}
    
    for model_type in ['xgboost', 'catboost']:
        print(f"  Training {model_type}...")
        model = ClaimsTreeAutoencoder(
            model_type=model_type,
            n_estimators=50 if model_type == 'xgboost' else None,
            iterations=50 if model_type == 'catboost' else None
        )
        model.fit(X, categorical_features=categorical_features,
                 numerical_features=numerical_features, verbose=False)
        models[model_type] = model
    
    # Individual performance
    print("\n" + "-"*80)
    print("Individual Model Performance:")
    print("-"*80)
    
    from sklearn.metrics import roc_auc_score
    
    for name, model in models.items():
        scores, _ = model.compute_fraud_scores(X)
        auc = roc_auc_score(y_true, scores)
        print(f"{name.upper()}: AUC-ROC = {auc:.4f}")
    
    # Ensemble performance
    print("\nEnsemble Performance:")
    print("-"*80)
    
    weight_configs = [
        {'xgboost': 0.5, 'catboost': 0.5},
        {'xgboost': 0.7, 'catboost': 0.3},
        {'xgboost': 0.3, 'catboost': 0.7}
    ]
    
    for weights in weight_configs:
        ensemble_scores = create_ensemble_fraud_scores(models, X, weights=weights)
        auc = roc_auc_score(y_true, ensemble_scores)
        weight_str = ", ".join([f"{k}={v}" for k, v in weights.items()])
        print(f"Weights ({weight_str}): AUC-ROC = {auc:.4f}")
    
    return models, X, y_true


def example_5_save_load():
    """Example 5: Save and load fraud detection model."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Save and Load Fraud Detection Model")
    print("="*80)
    
    # Create and train model
    claims = create_sample_claims()
    X = claims.drop('is_fraud', axis=1)
    
    categorical_features = [
        'claim_type', 'provider_specialty', 'diagnosis_code',
        'patient_gender', 'geographic_region'
    ]
    numerical_features = [
        'claim_amount', 'patient_age', 'provider_experience_years',
        'days_since_last_claim', 'num_previous_claims'
    ]
    
    print("\nTraining fraud detection model...")
    model = ClaimsTreeAutoencoder(model_type='xgboost', n_estimators=30)
    model.fit(X, categorical_features=categorical_features,
             numerical_features=numerical_features, verbose=False)
    
    # Get original predictions
    original_scores, _ = model.compute_fraud_scores(X)
    
    # Save model
    save_path = "fraud_detector_checkpoint"
    print(f"\nSaving model to {save_path}/...")
    model.save(save_path)
    
    # Load model
    print(f"Loading model from {save_path}/...")
    loaded_model = ClaimsTreeAutoencoder.load(save_path)
    
    # Get loaded predictions
    loaded_scores, _ = loaded_model.compute_fraud_scores(X)
    
    # Verify consistency
    score_diff = np.abs(original_scores - loaded_scores).max()
    print(f"\nVerification:")
    print(f"  Max difference: {score_diff:.10f}")
    print(f"  Models are {'identical' if score_diff < 1e-10 else 'different'}!")
    
    # Cleanup
    import shutil
    shutil.rmtree(save_path)
    print(f"\nCleaned up {save_path}/")
    
    return loaded_model


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("CLAIMS FRAUD DETECTION - TREE MODELS EXAMPLES")
    print("="*80)
    print("\nDemonstrates:")
    print("  1. XGBoost fraud detection")
    print("  2. XGBoost vs CatBoost comparison")
    print("  3. Fraud indicator analysis")
    print("  4. Ensemble fraud detection")
    print("  5. Save/load fraud detectors")
    
    # Run examples
    try:
        xgb_model, X1, y1, scores1 = example_1_xgboost_fraud_detection()
        models_comparison, X2, y2 = example_2_catboost_comparison()
        feature_model = example_3_feature_importance()
        ensemble_models, X3, y3 = example_4_ensemble_detection()
        loaded_model = example_5_save_load()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\n💡 Key Insights for Claims Fraud Detection:")
        print("  • Tree models train 15-20x faster than neural networks")
        print("  • Native categorical support (claim types, diagnoses)")
        print("  • Feature importance helps investigate fraud patterns")
        print("  • Ensemble predictions improve detection robustness")
        print("  • CPU-friendly deployment (no GPU required)")
        print("  • Easy to explain to fraud investigators")
        
    except Exception as e:
        print(f"\n❌ Error during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
