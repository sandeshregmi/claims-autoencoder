#!/bin/bash

# Quick Fix - Set up Python path and run the application

BASE_DIR="/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder"
cd "$BASE_DIR"

echo "Setting up Python path..."
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

echo "Starting application..."
echo "Access at: http://localhost:8501"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null || . venv/bin/activate 2>/dev/null
fi

streamlit run src/webapp_enhanced.py
