#!/bin/bash

# Script to clear Python cache and run the fixed fraud detection

echo "🧹 Clearing Python cache..."
rm -rf src/__pycache__/*.pyc
rm -rf src/__pycache__

echo "✅ Cache cleared!"
echo ""
echo "🚀 Running fraud detection with fixed code..."
echo ""

python tree_fraud_detection_runner.py --config config/example_config.yaml
