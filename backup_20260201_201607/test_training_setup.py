#!/usr/bin/env python3
"""
Test Training Setup

Quick script to verify the training pipeline works before running full training.
This helps identify any configuration or dependency issues.

Usage:
    python test_training_setup.py
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("="*80)
    print("Testing Imports...")
    print("="*80)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        from src.config_manager import ConfigManager
        print("✓ ConfigManager")
    except ImportError as e:
        print(f"✗ ConfigManager import failed: {e}")
        print(f"   Current sys.path: {sys.path}")
        return False
    
    try:
        from src.data_ingestion import DataIngestion
        print("✓ DataIngestion")
    except ImportError as e:
        print(f"✗ DataIngestion import failed: {e}")
        return False
    
    try:
        from src.preprocessing import ClaimsPreprocessor
        print("✓ ClaimsPreprocessor")
    except ImportError as e:
        print(f"✗ ClaimsPreprocessor import failed: {e}")
        return False
    
    try:
        from src.model_architecture import create_model_from_config
        print("✓ Model Architecture")
    except ImportError as e:
        print(f"✗ Model Architecture import failed: {e}")
        return False
    
    try:
        from src.training import ClaimsTrainer
        print("✓ ClaimsTrainer")
    except ImportError as e:
        print(f"✗ ClaimsTrainer import failed: {e}")
        return False
    
    # Test tree models
    try:
        from src.tree_models import ClaimsTreeAutoencoder
        print("✓ ClaimsTreeAutoencoder (Tree Models)")
    except ImportError as e:
        print(f"⚠ ClaimsTreeAutoencoder import failed (optional): {e}")
    
    return True


def test_config():
    """Test configuration loading."""
    print("\n" + "="*80)
    print("Testing Configuration...")
    print("="*80)
    
    try:
        from src.config_manager import ConfigManager
        
        config_path = "config/example_config.yaml"
        if not Path(config_path).exists():
            print(f"✗ Config file not found: {config_path}")
            return False
        
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        
        print(f"✓ Config loaded from {config_path}")
        print(f"  Train path: {config.data.train_path}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Max epochs: {config.training.max_epochs}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data():
    """Test data loading."""
    print("\n" + "="*80)
    print("Testing Data Loading...")
    print("="*80)
    
    try:
        import pandas as pd
        
        data_path = "data/claims_train.parquet"
        if not Path(data_path).exists():
            print(f"✗ Data file not found: {data_path}")
            print(f"  Please ensure {data_path} exists")
            return False
        
        df = pd.read_parquet(data_path)
        print(f"✓ Data loaded from {data_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns[:5])}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("\n" + "="*80)
    print("Testing Preprocessing...")
    print("="*80)
    
    try:
        import pandas as pd
        from src.config_manager import ConfigManager
        from src.preprocessing import ClaimsPreprocessor
        
        # Load config
        config_manager = ConfigManager("config/example_config.yaml")
        config = config_manager.get_config()
        
        # Load data
        df = pd.read_parquet("data/claims_train.parquet")
        
        # Take a small sample for testing
        df_sample = df.head(100)
        
        # Create preprocessor
        preprocessor = ClaimsPreprocessor(config)
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(df_sample)
        
        print(f"✓ Preprocessing successful")
        print(f"  Input shape: {df_sample.shape}")
        print(f"  Output shape: {X_transformed.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation."""
    print("\n" + "="*80)
    print("Testing Model Creation...")
    print("="*80)
    
    try:
        from src.config_manager import ConfigManager
        from src.model_architecture import create_model_from_config
        
        # Load config
        config_manager = ConfigManager("config/example_config.yaml")
        config = config_manager.get_config()
        
        # Create model
        input_dim = 50  # Example dimension
        model = create_model_from_config(config, input_dim)
        
        print(f"✓ Model created successfully")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Architecture: {model.get_layer_sizes()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CLAIMS AUTOENCODER - TRAINING SETUP TEST")
    print("="*80)
    print("\nThis script verifies that the training pipeline is properly configured.\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Data Loading", test_data),
        ("Preprocessing", test_preprocessing),
        ("Model Creation", test_model_creation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nYou can now run training with:")
        print("  python train.py --config config/example_config.yaml")
        print("\nOr try the tree models:")
        print("  python examples_tree_fraud_detection.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease fix the issues above before running training.")
        print("\nCommon solutions:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Check that data/claims_train.parquet exists")
        print("  3. Verify config/example_config.yaml is valid")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
