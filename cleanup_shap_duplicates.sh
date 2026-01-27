#!/bin/bash
# Clean up duplicate SHAP webapp versions

cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

echo "🧹 Cleaning up duplicate SHAP webapp versions..."
echo ""

# Check which files exist
echo "📋 Current files in src/:"
ls -lh src/webapp*.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Keep webapp_enhanced.py (the one you're using)
echo "✅ KEEPING: src/webapp_enhanced.py (your current version)"

# Remove duplicates
echo "🗑️  REMOVING duplicates:"

if [ -f "src/webapp_enhanced_with_shap.py" ]; then
    rm src/webapp_enhanced_with_shap.py
    echo "   ✅ Deleted: webapp_enhanced_with_shap.py"
fi

if [ -f "src/webapp_with_shap.py" ]; then
    rm src/webapp_with_shap.py
    echo "   ✅ Deleted: webapp_with_shap.py"
fi

if [ -f "src/webapp_complete_shap.py" ]; then
    rm src/webapp_complete_shap.py
    echo "   ✅ Deleted: webapp_complete_shap.py"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Remaining files in src/:"
ls -lh src/webapp*.py

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "You now have:"
echo "  • webapp.py (original basic version)"
echo "  • webapp_enhanced.py (your active version with SHAP)"
echo ""
echo "To run: streamlit run app_enhanced.py"
