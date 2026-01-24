#!/bin/bash

# Script to run all fraud detection models one by one
# Models: FT-Transformer (Neural), XGBoost, CatBoost

echo "================================================================================"
echo "COMPREHENSIVE FRAUD DETECTION MODEL COMPARISON"
echo "================================================================================"
echo ""
echo "This script will run:"
echo "  1. XGBoost (Tree Model) - ~5 seconds"
echo "  2. CatBoost (Tree Model) - ~5 seconds"  
echo "  3. FT-Transformer (Neural Network) - ~3-5 minutes"
echo ""
echo "Results will be saved to results/model_comparison/"
echo ""
echo "================================================================================"
echo ""

# Set environment for stability
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Create output directory
mkdir -p results/model_comparison
RESULTS_DIR="results/model_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================================"
echo "MODEL 1: XGBOOST (TREE-BASED)"
echo "================================================================================"
echo ""
echo "Starting XGBoost fraud detection..."
echo "Expected time: 5-10 seconds"
echo ""

START_TIME=$(date +%s)

# Clear cache first
rm -rf src/__pycache__

# Run XGBoost
python tree_fraud_detection_runner.py --config config/example_config.yaml --model xgboost | tee "$RESULTS_DIR/xgboost_${TIMESTAMP}.log"

END_TIME=$(date +%s)
XGB_TIME=$((END_TIME - START_TIME))

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ XGBoost completed successfully!"
    echo "   Training time: ${XGB_TIME} seconds"
    echo "   Results saved to: $RESULTS_DIR/xgboost_${TIMESTAMP}.log"
else
    echo "⚠️  XGBoost had issues - check log file"
fi

echo ""
echo "================================================================================"
echo "MODEL 2: CATBOOST (TREE-BASED)"
echo "================================================================================"
echo ""
echo "Starting CatBoost fraud detection..."
echo "Expected time: 5-10 seconds"
echo ""

START_TIME=$(date +%s)

# Clear cache first
rm -rf src/__pycache__

# Run CatBoost
python tree_fraud_detection_runner.py --config config/example_config.yaml --model catboost | tee "$RESULTS_DIR/catboost_${TIMESTAMP}.log"

END_TIME=$(date +%s)
CAT_TIME=$((END_TIME - START_TIME))

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ CatBoost completed successfully!"
    echo "   Training time: ${CAT_TIME} seconds"
    echo "   Results saved to: $RESULTS_DIR/catboost_${TIMESTAMP}.log"
else
    echo "⚠️  CatBoost had issues - check log file"
fi

echo ""
echo "================================================================================"
echo "MODEL 3: FT-TRANSFORMER (NEURAL NETWORK)"
echo "================================================================================"
echo ""
echo "Starting FT-Transformer training..."
echo "Expected time: 3-5 minutes (much slower but highest accuracy)"
echo ""
read -p "Do you want to run FT-Transformer? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    START_TIME=$(date +%s)

    # Run FT-Transformer
    python train.py --config config/example_config.yaml | tee "$RESULTS_DIR/ft_transformer_${TIMESTAMP}.log"

    END_TIME=$(date +%s)
    FT_TIME=$((END_TIME - START_TIME))

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ FT-Transformer completed successfully!"
        echo "   Training time: ${FT_TIME} seconds"
        echo "   Results saved to: $RESULTS_DIR/ft_transformer_${TIMESTAMP}.log"
    else
        echo "⚠️  FT-Transformer had issues - check log file"
    fi
else
    echo "Skipping FT-Transformer (Neural Network)"
    FT_TIME="skipped"
fi

echo ""
echo "================================================================================"
echo "COMPARISON SUMMARY"
echo "================================================================================"
echo ""
echo "Model Performance:"
echo "  XGBoost:         ${XGB_TIME} seconds"
echo "  CatBoost:        ${CAT_TIME} seconds"
echo "  FT-Transformer:  ${FT_TIME}"
echo ""
echo "Results Location: $RESULTS_DIR/"
echo ""
echo "To view results:"
echo "  XGBoost:        cat $RESULTS_DIR/xgboost_${TIMESTAMP}.log | tail -50"
echo "  CatBoost:       cat $RESULTS_DIR/catboost_${TIMESTAMP}.log | tail -50"
if [[ $FT_TIME != "skipped" ]]; then
    echo "  FT-Transformer: cat $RESULTS_DIR/ft_transformer_${TIMESTAMP}.log | tail -50"
fi
echo ""
echo "================================================================================"
echo "ALL MODELS COMPLETE!"
echo "================================================================================"
