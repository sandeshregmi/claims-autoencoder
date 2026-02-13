#!/bin/bash

# GitHub Actions Status Checker
# This script helps diagnose why GitHub Actions might not be running

echo "🔍 GitHub Actions Diagnostic Check"
echo "===================================="
echo ""

# Check if .github/workflows directory exists
if [ -d ".github/workflows" ]; then
    echo "✅ Workflow directory exists"
    echo ""
    echo "📋 Workflow files found:"
    ls -1 .github/workflows/
    echo ""
else
    echo "❌ No .github/workflows directory found"
    exit 1
fi

# Check if CI workflow exists
if [ -f ".github/workflows/ci.yml" ]; then
    echo "✅ CI workflow file exists"
    echo ""
    
    # Check workflow triggers
    echo "🔔 Workflow Triggers:"
    grep -A 5 "^on:" .github/workflows/ci.yml | head -6
    echo ""
else
    echo "❌ CI workflow file not found"
fi

# Check recent git commits
echo "📝 Recent Commits (last 5):"
git log --oneline -5
echo ""

# Check git remote
echo "🌐 Git Remote:"
git remote -v | head -2
echo ""

# Check current branch
echo "🌳 Current Branch:"
git branch -v
echo ""

# Check if there are uncommitted changes
echo "📊 Git Status:"
git status --short || echo "Working tree is clean"
echo ""

echo "🎯 Next Steps:"
echo "1. Verify Actions are enabled: https://github.com/sandeshregmi/claims-autoencoder/settings/actions"
echo "2. Check workflow runs: https://github.com/sandeshregmi/claims-autoencoder/actions"
echo "3. Make a new commit to trigger the workflow:"
echo "   git commit --allow-empty -m 'Trigger CI workflow'"
echo "   git push origin main"
