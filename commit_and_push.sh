#!/bin/bash

echo "🚀 Committing and Pushing Enhanced Configuration v2.0"
echo "======================================================"
echo ""

cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

echo "1️⃣  Adding all changes..."
git add -A
echo "   ✅ Changes staged"
echo ""

echo "2️⃣  Creating commit..."
git commit -m "feat: Enhanced configuration v2.0

Major improvements:
- ✅ Fully configurable system (95%+ coverage)
- ✅ Data validation module with auto-fix
- ✅ Business rules engine for fraud detection
- ✅ Fixed Streamlit caching (DictConfig now hashable)
- ✅ All hardcoded values moved to config
- ✅ Feature schemas with validation rules
- ✅ Categorical domains with allowed values
- ✅ Business rules in YAML config
- ✅ GitHub Actions workflows configured

Files changed:
- src/claims_fraud/config/manager.py (hashable DictConfig)
- config/config.yaml (comprehensive configuration)
- src/claims_fraud/data/validation.py (data validator)
- src/claims_fraud/core/business_rules.py (business rules)
- Documentation updates

Breaking changes: None
Migration guide: See IMPLEMENTATION_SUCCESS.md
"

COMMIT_STATUS=$?

if [ $COMMIT_STATUS -eq 0 ]; then
    echo "   ✅ Commit created"
else
    echo "   ⚠️  Commit failed or nothing to commit"
fi
echo ""

echo "3️⃣  Pushing to GitHub..."
git push origin main

PUSH_STATUS=$?

if [ $PUSH_STATUS -eq 0 ]; then
    echo "   ✅ Pushed to GitHub!"
    echo ""
    echo "======================================================"
    echo "🎉 SUCCESS!"
    echo "======================================================"
    echo ""
    echo "Your GitHub Actions should now run!"
    echo ""
    echo "Check them here:"
    echo "https://github.com/sandeshregmi/claims-autoencoder/actions"
    echo ""
    echo "Expected workflows:"
    echo "  ✅ Build Check (runs on all branches)"
    echo "  ✅ CI - Tests and Validation (runs on main/develop)"
    echo ""
else
    echo "   ❌ Push failed"
    echo ""
    echo "Possible issues:"
    echo "  - No internet connection"
    echo "  - Authentication required"
    echo "  - Branch protection rules"
    echo ""
    echo "Try manually:"
    echo "  git push origin main"
fi
