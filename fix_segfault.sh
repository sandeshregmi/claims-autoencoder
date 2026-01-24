#!/bin/bash

# Quick fix for XGBoost segmentation fault on macOS
# Often caused by OpenMP threading issues

echo "🔧 Attempting to fix XGBoost segmentation fault..."
echo ""

# Method 1: Limit threads
echo "Method 1: Setting single thread mode..."
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Environment variables set:"
echo "  OMP_NUM_THREADS=1"
echo "  OPENBLAS_NUM_THREADS=1"
echo "  MKL_NUM_THREADS=1"
echo ""

echo "🧪 Running diagnostic test..."
python test_xgboost.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Diagnostic passed! Now trying fraud detection..."
    echo ""
    python tree_fraud_detection_runner.py --config config/example_config.yaml
else
    echo ""
    echo "❌ Diagnostic failed. Try:"
    echo "   pip install --upgrade --force-reinstall xgboost"
fi
