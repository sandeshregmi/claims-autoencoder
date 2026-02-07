#!/bin/bash

# Quick Fix - Install Missing Dependencies

echo "Installing missing dependencies..."
pip install seaborn>=0.12.0 matplotlib>=3.7.0 --quiet

echo "✅ Dependencies installed!"
echo ""
echo "You can now run the application:"
echo "  ./run_clean_workflow.sh"
echo ""
echo "Or start it manually:"
echo "  streamlit run src/webapp_enhanced.py"
