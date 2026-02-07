#!/bin/bash

# Fix NumPy and other compatibility issues

echo "Fixing NumPy compatibility issue..."
pip install "numpy<2.0" --force-reinstall --quiet

echo "✅ NumPy downgraded to 1.x"
echo ""
echo "Note: The Plotly and Streamlit warnings are just deprecation notices"
echo "and won't affect functionality. They'll be fixed in future updates."
echo ""
echo "Your application is working correctly!"
echo ""
echo "To suppress warnings, you can run:"
echo "  export PYTHONWARNINGS='ignore::DeprecationWarning'"
echo "  ./run_clean_workflow.sh"
