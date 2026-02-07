#!/bin/bash

# Clean Workflow Runner - Suppresses Deprecation Warnings

set -e

BASE_DIR="/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder"
cd "$BASE_DIR"

echo "=========================================="
echo "Claims Fraud Detection - Clean Workflow"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate || . venv/bin/activate
    echo "✓ Virtual environment activated"
    echo ""
fi

# Set PYTHONPATH
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Suppress deprecation warnings (optional)
export PYTHONWARNINGS='ignore::DeprecationWarning'

echo "Starting web application (warnings suppressed)..."
echo ""
echo "Access the dashboard at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run src/webapp_enhanced.py
