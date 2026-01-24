#!/bin/bash
# Quick script to run XGBoost fraud detection

echo "🚀 Running XGBoost Fraud Detection..."
echo ""

# Set environment
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Clear cache
rm -rf src/__pycache__

# Run XGBoost
python tree_fraud_detection_runner.py --config config/example_config.yaml --model xgboost
