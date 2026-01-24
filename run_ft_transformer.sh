#!/bin/bash
# Quick script to run FT-Transformer (Neural Network)

echo "🚀 Running FT-Transformer (Neural Network)..."
echo "⏱️  This will take 3-5 minutes..."
echo ""

# Set environment
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run FT-Transformer
python train.py --config config/example_config.yaml
