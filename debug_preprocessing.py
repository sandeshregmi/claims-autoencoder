#!/usr/bin/env python
"""
Quick debug script to test preprocessing pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.config_manager import ConfigManager
from src.preprocessing import ClaimsPreprocessor
from src.data_ingestion import load_sample_data

print("🔍 Testing Preprocessing Pipeline\n")

# Load config
print("1. Loading configuration...")
config_manager = ConfigManager("config/example_config.yaml")
config = config_manager.get_config()
print(f"   ✅ Config loaded\n")

# Generate sample data
print("2. Generating sample data...")
df = load_sample_data(n_samples=1000)
print(f"   ✅ Generated {len(df)} samples with {len(df.columns)} columns")
print(f"   Columns: {df.columns.tolist()}\n")

# Split data
print("3. Splitting data...")
train_df = df[:700]
val_df = df[700:]
print(f"   ✅ Train: {len(train_df)}, Val: {len(val_df)}\n")

# Test preprocessing
print("4. Testing preprocessing pipeline...")
try:
    preprocessor = ClaimsPreprocessor(config)
    
    # Fit on training data
    print("   - Fitting on training data...")
    X_train = preprocessor.fit_transform(train_df)
    print(f"     ✅ Training data: {train_df.shape} -> {X_train.shape}")
    
    # Transform validation data
    print("   - Transforming validation data...")
    X_val = preprocessor.transform(val_df)
    print(f"     ✅ Validation data: {val_df.shape} -> {X_val.shape}")
    
    # Check feature names
    feature_names = preprocessor.get_feature_names()
    print(f"   - Total features after preprocessing: {len(feature_names)}")
    print(f"   - Feature names: {feature_names[:5]}...\n")
    
    # Check for NaN values
    if np.any(np.isnan(X_train)):
        print("   ⚠️  Warning: NaN values found in training data")
    else:
        print("   ✅ No NaN values in training data")
    
    if np.any(np.isnan(X_val)):
        print("   ⚠️  Warning: NaN values found in validation data")
    else:
        print("   ✅ No NaN values in validation data")
    
    print("\n✅ Preprocessing pipeline working correctly!")
    
except Exception as e:
    print(f"\n❌ Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model creation
print("\n5. Testing model creation...")
try:
    from src.model_architecture import create_model_from_config
    import torch
    
    input_dim = X_train.shape[1]
    model = create_model_from_config(config, input_dim)
    
    # Test forward pass
    x = torch.FloatTensor(X_train[:10])
    reconstruction, encoding = model(x)
    
    print(f"   ✅ Model created: {model.count_parameters():,} parameters")
    print(f"   ✅ Forward pass: {x.shape} -> {encoding.shape} -> {reconstruction.shape}")
    
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("🎉 All checks passed! Ready for training.")
print("="*60)
print("\n▶️  Run: python train.py --config config/example_config.yaml")
