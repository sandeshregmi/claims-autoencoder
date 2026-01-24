#!/bin/bash
# Quick script to run CatBoost fraud detection

echo "🚀 Running CatBoost Fraud Detection..."
echo ""

# Set environment
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Clear cache
rm -rf src/__pycache__

# Run CatBoost
python tree_fraud_detection_runner.py --config config/example_config.yaml --model catboost
