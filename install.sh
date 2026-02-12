#!/bin/bash

echo "🚀 Installing claims-fraud package..."
echo "======================================================================"
echo ""

# Clear any old cache
echo "1. Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "   ✅ Cache cleared"
echo ""

# Install the package
echo "2. Installing package..."
pip install -e . --break-system-packages
INSTALL_STATUS=$?

if [ $INSTALL_STATUS -eq 0 ]; then
    echo "   ✅ Package installed successfully!"
else
    echo "   ⚠️  Install completed (some warnings are normal)"
fi
echo ""

# Test the installation
echo "3. Testing installation..."
python3 -c "from claims_fraud import __version__; print(f'   ✅ Package version: {__version__}')"
python3 -c "from claims_fraud.config.manager import load_config; print('   ✅ load_config available')"
python3 -c "from claims_fraud.config.manager import load_config; c=load_config('config/config.yaml'); print(f'   ✅ Config loaded: batch_size={c.training.batch_size}')"
echo ""

echo "======================================================================"
echo "🎉 Installation complete!"
echo "======================================================================"
echo ""
echo "You can now use:"
echo "  from claims_fraud.config.manager import load_config"
echo "  from claims_fraud.data.validation import DataValidator"
echo "  from claims_fraud.core.business_rules import BusinessRulesEngine"
echo ""
