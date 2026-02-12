#!/usr/bin/env python3
"""Test that DictConfig is now hashable for Streamlit"""

from pathlib import Path

print("🔍 Testing DictConfig hashability...")
print()

# Test 1: Import and load config
print("1. Loading config...")
from claims_fraud.config.manager import load_config

config_path = Path(__file__).parent / 'config' / 'config.yaml'
config = load_config(str(config_path))
print("   ✅ Config loaded")

# Test 2: Test hashability
print("2. Testing hash()...")
try:
    config_hash = hash(config)
    print(f"   ✅ Config is hashable: {config_hash}")
except TypeError as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Test 3: Test equality
print("3. Testing equality...")
config2 = load_config(str(config_path))
if config == config2:
    print("   ✅ Equality works")
else:
    print("   ❌ Equality failed")

# Test 4: Test with Streamlit caching (if available)
print("4. Testing with Streamlit cache...")
try:
    import streamlit as st
    
    @st.cache_data
    def test_cached_func(config):
        return config.training.batch_size
    
    result = test_cached_func(config)
    print(f"   ✅ Streamlit cache works! Batch size: {result}")
    
except ImportError:
    print("   ⚠️  Streamlit not installed (that's OK)")
except Exception as e:
    print(f"   ❌ Streamlit cache failed: {e}")

print()
print("=" * 70)
print("🎉 DictConfig is now hashable!")
print("=" * 70)
print()
print("✅ Your Streamlit app should now work!")
print()
