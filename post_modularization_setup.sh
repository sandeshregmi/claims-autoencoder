#!/bin/bash
# Post-Modularization Setup Script
# Run this after modularization completes

echo "=============================================="
echo "Claims Fraud Detection - Post-Setup"
echo "=============================================="
echo ""

# Step 1: Install the package
echo "Step 1: Installing package..."
echo ""
echo "Since you're in conda base environment, run:"
echo ""
echo "  cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder"
echo "  pip install -e ."
echo ""
read -p "Press Enter after running the above command..."

# Step 2: Verify installation
echo ""
echo "Step 2: Verifying installation..."
echo ""

# Check CLI
if command -v claims-fraud &> /dev/null; then
    echo "✅ CLI installed successfully"
    claims-fraud --version
else
    echo "❌ CLI not found - installation may have failed"
    echo "   Try: pip install -e . --force-reinstall"
fi

# Check Python import
echo ""
echo "Testing Python import..."
python -c "import claims_fraud; print('✅ Import successful! Version:', claims_fraud.__version__)" 2>&1

# Step 3: Test CLI commands
echo ""
echo "Step 3: Testing CLI commands..."
echo ""

claims-fraud --help

# Step 4: Show next steps
echo ""
echo "=============================================="
echo "✅ Setup Complete!"
echo "=============================================="
echo ""
echo "📋 What to do next:"
echo ""
echo "1. Test the dashboard:"
echo "   claims-fraud serve"
echo ""
echo "2. Try the CLI:"
echo "   claims-fraud --help"
echo "   claims-fraud train --help"
echo ""
echo "3. Test Python API:"
echo "   python -c 'from claims_fraud import FraudDetector, TreeModel'"
echo ""
echo "4. Run examples:"
echo "   python examples/quickstart.py"
echo ""
echo "5. Run tests:"
echo "   pytest"
echo ""
echo "=============================================="
