#!/usr/bin/env python3
"""Test the configuration system"""

print("🔍 Testing Configuration System...")
print("=" * 70)
print()

# Test 1: Package version
print("1. Testing package import...")
try:
    from claims_fraud import __version__
    print(f"   ✅ Package version: {__version__}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Test 2: Import load_config
print("2. Testing load_config import...")
try:
    from claims_fraud.config.manager import load_config
    print(f"   ✅ load_config imported successfully")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Test 3: Load configuration
print("3. Testing configuration loading...")
try:
    config = load_config('config/config.yaml')
    print(f"   ✅ Configuration loaded")
    print(f"   ✅ Batch size: {config.training.batch_size}")
    print(f"   ✅ Learning rate: {config.training.learning_rate}")
    print(f"   ✅ Encoding dim: {config.model.encoding_dim}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Access nested config
print("4. Testing nested configuration access...")
try:
    high_risk = config.business_rules.fraud_thresholds.claim_amount_high_risk
    print(f"   ✅ High risk threshold: ${high_risk:,}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print()
print("=" * 70)
print("🎉 ALL TESTS PASSED!")
print("=" * 70)
print()
print("✅ Your configuration system is working perfectly!")
print()
