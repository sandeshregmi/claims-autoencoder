#!/usr/bin/env python3
"""
Simple test script to run from terminal
Run from: /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/
"""

import os
from pathlib import Path

print("🔍 Testing claims_fraud package from terminal...")
print("=" * 70)
print()

# Show current directory
print(f"Current directory: {os.getcwd()}")
print()

# Test 1: Import package
print("1. Testing package import...")
try:
    from claims_fraud import __version__
    print(f"   ✅ Package version: {__version__}")
except ImportError as e:
    print(f"   ❌ Failed: {e}")
    print("   💡 Make sure you're in: /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/")
    exit(1)

# Test 2: Import load_config
print("2. Testing load_config import...")
try:
    from claims_fraud.config.manager import load_config
    print(f"   ✅ load_config imported")
except ImportError as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Test 3: Load config with full path
print("3. Testing config loading...")
try:
    # Use full absolute path
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    print(f"   Config path: {config_path}")
    
    if not config_path.exists():
        print(f"   ❌ Config file not found at: {config_path}")
        print(f"   💡 Make sure config/config.yaml exists")
        exit(1)
    
    config = load_config(str(config_path))
    print(f"   ✅ Config loaded successfully")
    print(f"   ✅ Batch size: {config.training.batch_size}")
    print(f"   ✅ Learning rate: {config.training.learning_rate}")
    print(f"   ✅ High risk threshold: ${config.business_rules.fraud_thresholds.claim_amount_high_risk:,}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("=" * 70)
print("🎉 ALL TESTS PASSED!")
print("=" * 70)
print()
print("✅ You can now use the package!")
print()
print("Example usage:")
print("""
from claims_fraud.config.manager import load_config
config = load_config('config/config.yaml')
print(f"Batch size: {config.training.batch_size}")
""")
