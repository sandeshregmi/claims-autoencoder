#!/usr/bin/env python
"""
Quick test to verify the setup is working correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("🔍 Checking Claims Autoencoder Setup...\n")

# Test 1: Check imports
print("1️⃣ Testing imports...")
try:
    from src.config_manager import ConfigManager
    from src.data_ingestion import DataIngestion, load_sample_data
    from src.preprocessing import ClaimsPreprocessor
    from src.model_architecture import ClaimsAutoencoder
    print("   ✅ All imports successful!\n")
except Exception as e:
    print(f"   ❌ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Check configuration
print("2️⃣ Testing configuration...")
try:
    config_manager = ConfigManager("config/example_config.yaml")
    config = config_manager.get_config()
    print(f"   ✅ Config loaded: {config.data.train_path}\n")
except Exception as e:
    print(f"   ❌ Config failed: {e}\n")
    sys.exit(1)

# Test 3: Check data
print("3️⃣ Checking data files...")
data_path = Path("data/claims_train.parquet")
if data_path.exists():
    import pandas as pd
    df = pd.read_parquet(data_path)
    print(f"   ✅ Training data found: {len(df)} rows, {len(df.columns)} columns")
    print(f"   📊 Columns: {', '.join(df.columns.tolist()[:5])}...\n")
else:
    print(f"   ⚠️  Training data not found at {data_path}")
    print("   💡 Generate sample data with: make sample-data\n")

# Test 4: Check directories
print("4️⃣ Checking directories...")
dirs_to_check = ["data", "models", "outputs", "logs", "checkpoints"]
missing_dirs = []
for dir_name in dirs_to_check:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"   ✅ {dir_name}/")
    else:
        print(f"   ⚠️  {dir_name}/ (will be created)")
        missing_dirs.append(dir_name)

if missing_dirs:
    print(f"\n   💡 Create missing directories with: mkdir -p {' '.join(missing_dirs)}\n")
else:
    print()

# Test 5: Test model creation
print("5️⃣ Testing model creation...")
try:
    import torch
    model = ClaimsAutoencoder(
        input_dim=20,
        encoding_dim=32,
        hidden_layers=[128, 64],
        dropout_rate=0.3
    )
    x = torch.randn(4, 20)
    reconstruction, encoding = model(x)
    print(f"   ✅ Model created with {model.count_parameters():,} parameters")
    print(f"   ✅ Forward pass successful: {x.shape} -> {encoding.shape} -> {reconstruction.shape}\n")
except Exception as e:
    print(f"   ❌ Model test failed: {e}\n")
    sys.exit(1)

# Test 6: Test preprocessing
print("6️⃣ Testing preprocessing...")
try:
    if data_path.exists():
        df = pd.read_parquet(data_path)
        preprocessor = ClaimsPreprocessor(config)
        X = preprocessor.fit_transform(df[:100])
        print(f"   ✅ Preprocessor working: {len(df[:100])} samples -> {X.shape}\n")
    else:
        print("   ⏭️  Skipped (no data file)\n")
except Exception as e:
    print(f"   ❌ Preprocessing failed: {e}\n")

# Summary
print("="*50)
print("📋 SETUP SUMMARY")
print("="*50)

if data_path.exists():
    print("✅ Ready to train!")
    print("\n🚀 Run: python train.py --config config/example_config.yaml")
else:
    print("⚠️  Almost ready! Just need data.")
    print("\n📦 Generate sample data:")
    print("   make sample-data")
    print("   # OR")
    print("   python -c \"import sys; sys.path.insert(0, '.'); from src.data_ingestion import load_sample_data; df = load_sample_data(10000); df.to_parquet('data/claims_train.parquet')\"")
    print("\n🚀 Then run: python train.py --config config/example_config.yaml")

print("\n📚 Documentation:")
print("   • SOLUTION.md - Troubleshooting guide")
print("   • QUICKSTART.md - Quick reference")
print("   • README.md - Full documentation")
print("="*50)
