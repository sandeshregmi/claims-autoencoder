"""Quickstart example for Claims Fraud Detection"""
from claims_fraud import TreeModel, FraudDetector
import pandas as pd

def main():
    """Run quickstart example"""
    print("🚀 Claims Fraud Detection - Quickstart")
    print("="*50)
    
    # 1. Load data
    print("
1. Loading sample data...")
    data = pd.read_parquet("data/sample_claims.parquet")
    print(f"   Loaded {len(data)} claims")
    
    # 2. Create model
    print("
2. Creating model...")
    model = TreeModel(model_type="catboost")
    
    # 3. Train
    print("
3. Training model...")
    model.fit(
        data,
        categorical_features=['claim_type', 'patient_gender'],
        numerical_features=['claim_amount', 'patient_age']
    )
    print("   ✅ Training complete!")
    
    # 4. Create detector
    print("
4. Creating fraud detector...")
    detector = FraudDetector(model)
    
    # 5. Score claims
    print("
5. Scoring claims...")
    scores = detector.predict(data)
    
    # 6. Analyze
    print("
6. Analyzing results...")
    threshold = scores.quantile(0.95)
    high_risk = scores > threshold
    
    print(f"   Total claims: {len(scores)}")
    print(f"   High risk: {high_risk.sum()} ({high_risk.sum()/len(scores)*100:.1f}%)")
    print(f"   Threshold: {threshold:,.0f}")
    
    # 7. Save model
    print("
7. Saving model...")
    detector.save("models/quickstart_model.pkl")
    print("   ✅ Model saved!")
    
    print("
✅ Quickstart complete!")
    print("
Next steps:")
    print("  - Try: claims-fraud serve")
    print("  - See: examples/batch_scoring.py")

if __name__ == "__main__":
    main()
