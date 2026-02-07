#!/usr/bin/env python
"""
Quick diagnostic script to test XGBoost installation
"""

import sys
import pandas as pd
import numpy as np

print("=" * 60)
print("XGBOOST DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: Import check
print("\n1. Testing XGBoost import...")
try:
    import xgboost as xgb
    print(f"   ✅ XGBoost version: {xgb.__version__}")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Simple regression
print("\n2. Testing simple XGBoost regression...")
try:
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
    model.fit(X, y, verbose=False)
    pred = model.predict(X)
    print(f"   ✅ Simple regression works")
except Exception as e:
    print(f"   ❌ Regression failed: {e}")
    sys.exit(1)

# Test 3: Simple classification
print("\n3. Testing simple XGBoost classification...")
try:
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
    model.fit(X, y, verbose=False)
    pred = model.predict(X)
    print(f"   ✅ Simple classification works")
except Exception as e:
    print(f"   ❌ Classification failed: {e}")
    sys.exit(1)

# Test 4: DataFrame input
print("\n4. Testing with DataFrame input...")
try:
    df = pd.DataFrame(np.random.rand(100, 5), columns=['a', 'b', 'c', 'd', 'e'])
    y = pd.Series(np.random.rand(100))
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
    model.fit(df, y, verbose=False)
    pred = model.predict(df)
    print(f"   ✅ DataFrame input works")
except Exception as e:
    print(f"   ❌ DataFrame test failed: {e}")
    sys.exit(1)

# Test 5: Categorical encoding
print("\n5. Testing categorical encoding...")
try:
    df = pd.DataFrame({
        'num1': np.random.rand(100),
        'num2': np.random.rand(100),
        'cat1': np.random.choice(['A', 'B', 'C'], 100),
        'cat2': np.random.choice(['X', 'Y', 'Z'], 100)
    })
    
    # Convert to codes (like our fix does)
    df_encoded = df.copy()
    df_encoded['cat1'] = pd.Categorical(df['cat1']).codes
    df_encoded['cat2'] = pd.Categorical(df['cat2']).codes
    
    y = pd.Series(np.random.rand(100))
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
    model.fit(df_encoded, y, verbose=False)
    pred = model.predict(df_encoded)
    print(f"   ✅ Categorical encoding works")
except Exception as e:
    print(f"   ❌ Categorical test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Load real data
print("\n6. Testing with real claims data...")
try:
    df = pd.read_parquet('data/claims_train.parquet')
    print(f"   ✅ Loaded {len(df)} claims")
    print(f"   Columns: {list(df.columns)}")
    
    # Check for NaN
    nan_count = df.isna().sum().sum()
    print(f"   NaN values: {nan_count}")
    
    if nan_count > 0:
        print("\n   NaN values by column:")
        for col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                print(f"     {col}: {n} ({n/len(df)*100:.1f}%)")
    
except Exception as e:
    print(f"   ❌ Data load failed: {e}")
    sys.exit(1)

# Test 7: Train on one feature (like the autoencoder does)
print("\n7. Testing autoencoder-style training (single feature as target)...")
try:
    df = pd.read_parquet('data/claims_train.parquet')
    
    # Pick a numerical feature as target
    target_col = 'claim_amount'
    predictor_cols = [c for c in df.columns if c != target_col]
    
    X = df[predictor_cols].head(100)  # Just 100 rows for speed
    y = df[target_col].head(100)
    
    # Check for NaN in target
    if y.isna().any():
        print(f"   ⚠️  Target has {y.isna().sum()} NaN values - filtering...")
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
    
    # Encode categorical columns
    cat_cols = ['claim_type', 'provider_specialty', 'diagnosis_code', 
                'procedure_code', 'patient_gender', 'geographic_region']
    
    X_encoded = X.copy()
    for col in cat_cols:
        if col in X_encoded.columns:
            codes = pd.Categorical(X_encoded[col]).codes
            X_encoded[col] = np.where(codes == -1, 0, codes)
    
    # Handle NaN in predictors
    for col in X_encoded.columns:
        if X_encoded[col].isna().any():
            median_val = X_encoded[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X_encoded[col] = X_encoded[col].fillna(median_val)
    
    print(f"   Training on {len(X_encoded)} samples...")
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
    model.fit(X_encoded, y, verbose=False)
    pred = model.predict(X_encoded)
    print(f"   ✅ Autoencoder-style training works!")
    
except Exception as e:
    print(f"   ❌ Autoencoder test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nXGBoost installation is working correctly.")
print("The segmentation fault is likely coming from something else.")
print("\nTry:")
print("1. Update XGBoost: pip install --upgrade xgboost")
print("2. Check system libraries: otool -L $(python -c 'import xgboost; print(xgboost.__file__.replace(\"__init__.py\", \"lib/libxgboost.dylib\"))')")
print("3. Run with fewer threads: export OMP_NUM_THREADS=1")
