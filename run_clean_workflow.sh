#!/bin/bash

# Clean Workflow Pipeline Runner
# Runs the end-to-end claims fraud detection pipeline

set -e

BASE_DIR="/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder"
cd "$BASE_DIR"

echo "=========================================="
echo "Claims Fraud Detection - Clean Workflow"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || . venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install/update dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements_clean.txt
echo "✓ Dependencies installed"
echo ""

# Check if data exists
if [ ! -f "data/claims_train.parquet" ]; then
    echo "ERROR: Training data not found at data/claims_train.parquet"
    echo "Please ensure your data file is in the correct location."
    exit 1
fi
echo "✓ Data file found"
echo ""

# Check if config exists
if [ ! -f "config/starter_config.yaml" ]; then
    echo "ERROR: Configuration file not found at config/starter_config.yaml"
    exit 1
fi
echo "✓ Configuration file found"
echo ""

# Set PYTHONPATH to include the project root
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
echo "✓ Python path configured"
echo ""

# Run the web application
echo "Starting web application..."
echo ""
echo "Access the dashboard at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
streamlit run src/webapp_enhanced.py
