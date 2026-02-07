#!/bin/bash

#############################################
# Claims Fraud Detection - Main Runner
# Clean output with warnings suppressed
#############################################

set -e

BASE_DIR="/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder"
cd "$BASE_DIR"

echo "=========================================="
echo "Claims Fraud Detection Dashboard"
echo "=========================================="
echo ""

# Step 1: Setup virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Step 2: Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate || . venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Step 3: Install/Update dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip

# Fix NumPy compatibility first
echo "  → Fixing NumPy compatibility..."
pip install -q "numpy>=1.24.0,<2.0" --force-reinstall

# Install all other dependencies
echo "  → Installing packages..."
pip install -q -r requirements_clean.txt

echo "✓ All dependencies installed"
echo ""

# Step 4: Verify data and config
if [ ! -f "data/claims_train.parquet" ]; then
    echo "❌ ERROR: Training data not found at data/claims_train.parquet"
    echo "Please add your data file to the data/ directory."
    exit 1
fi
echo "✓ Data file found"

if [ ! -f "config/starter_config.yaml" ]; then
    echo "❌ ERROR: Configuration file not found"
    exit 1
fi
echo "✓ Configuration file found"
echo ""

# Step 5: Set Python path
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
echo "✓ Python path configured"
echo ""

# Step 6: Suppress warnings for clean output
export PYTHONWARNINGS='ignore::DeprecationWarning,ignore::UserWarning,ignore::FutureWarning'

# Step 7: Launch application
echo "=========================================="
echo "🚀 Starting Dashboard (clean output)..."
echo "=========================================="
echo ""
echo "📍 Access the dashboard at:"
echo "   → http://localhost:8501"
echo ""
echo "💡 Press Ctrl+C to stop"
echo ""
echo "ℹ️  Deprecation warnings suppressed for"
echo "   cleaner output. Everything works fine!"
echo ""
echo "=========================================="
echo ""

streamlit run src/webapp_enhanced.py 2>&1 | grep -v "deprecated\|deprecat" | grep -v "keyword arguments"
