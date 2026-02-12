#!/bin/bash

echo "🔍 GitHub Actions Diagnostic Tool"
echo "=================================="
echo ""

# Change to project directory
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder 2>/dev/null || {
    echo "❌ Cannot find project directory"
    echo "   Expected: /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder"
    exit 1
}

echo "✅ Project directory found"
echo ""

# Check if git repo
echo "1️⃣  Checking Git Repository..."
if [ ! -d ".git" ]; then
    echo "   ❌ NOT a git repository!"
    echo "   💡 Run: git init && git remote add origin https://github.com/sandeshregmi/claims-autoencoder.git"
    exit 1
fi
echo "   ✅ Git repository found"
echo ""

# Check current branch
echo "2️⃣  Current Branch:"
BRANCH=$(git branch --show-current 2>/dev/null)
if [ -z "$BRANCH" ]; then
    echo "   ❌ No branch checked out"
else
    echo "   ✅ $BRANCH"
    if [ "$BRANCH" != "main" ] && [ "$BRANCH" != "develop" ]; then
        echo "   ⚠️  WARNING: Workflows trigger on 'main' or 'develop'"
        echo "   💡 Your current branch: $BRANCH"
    fi
fi
echo ""

# Check if workflows exist and are tracked
echo "3️⃣  Checking Workflows..."
if [ ! -d ".github/workflows" ]; then
    echo "   ❌ .github/workflows directory not found"
else
    TRACKED=$(git ls-files .github/workflows/ | wc -l)
    ACTUAL=$(find .github/workflows -name "*.yml" -o -name "*.yaml" | wc -l)
    
    echo "   Total workflow files: $ACTUAL"
    echo "   Tracked by git: $TRACKED"
    
    if [ "$TRACKED" -eq 0 ]; then
        echo "   ❌ Workflows exist but NOT tracked by git!"
        echo "   💡 Fix: git add .github/workflows/"
    elif [ "$TRACKED" -lt "$ACTUAL" ]; then
        echo "   ⚠️  Some workflows not tracked"
        echo "   💡 Fix: git add .github/workflows/"
    else
        echo "   ✅ All workflows tracked by git"
        git ls-files .github/workflows/ | while read file; do
            echo "      - $file"
        done
    fi
fi
echo ""

# Check git status
echo "4️⃣  Git Status:"
STATUS=$(git status --short | wc -l)
if [ "$STATUS" -eq 0 ]; then
    echo "   ✅ Working directory clean"
else
    echo "   ⚠️  You have uncommitted changes:"
    git status --short | head -10 | sed 's/^/      /'
    if [ "$STATUS" -gt 10 ]; then
        echo "      ... and $((STATUS - 10)) more"
    fi
fi
echo ""

# Check if ahead of remote
echo "5️⃣  Checking Remote Sync..."
git fetch origin --quiet 2>/dev/null
AHEAD=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo "0")
BEHIND=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo "0")

if [ "$AHEAD" -gt 0 ]; then
    echo "   ⚠️  You are $AHEAD commit(s) ahead of origin/main"
    echo "   💡 Push to trigger workflows: git push origin main"
elif [ "$BEHIND" -gt 0 ]; then
    echo "   ⚠️  You are $BEHIND commit(s) behind origin/main"
    echo "   💡 Pull latest: git pull origin main"
else
    echo "   ✅ In sync with remote"
fi
echo ""

# Check recent commits
echo "6️⃣  Recent Commits:"
git log --oneline -3 | sed 's/^/      /'
echo ""

# Summary and recommendations
echo "=================================="
echo "📋 SUMMARY & RECOMMENDATIONS"
echo "=================================="
echo ""

# Determine what needs to be done
NEEDS_ADD=false
NEEDS_COMMIT=false
NEEDS_PUSH=false

if [ "$TRACKED" -eq 0 ] && [ "$ACTUAL" -gt 0 ]; then
    NEEDS_ADD=true
fi

if [ "$STATUS" -gt 0 ]; then
    NEEDS_COMMIT=true
fi

if [ "$AHEAD" -gt 0 ]; then
    NEEDS_PUSH=true
fi

if [ "$NEEDS_ADD" = true ]; then
    echo "⚠️  Step 1: Track workflows"
    echo "   git add .github/workflows/"
    echo ""
fi

if [ "$NEEDS_COMMIT" = true ]; then
    echo "⚠️  Step 2: Commit changes"
    echo "   git add -A"
    echo "   git commit -m \"feat: Enhanced configuration v2.0\""
    echo ""
fi

if [ "$NEEDS_PUSH" = true ]; then
    echo "⚠️  Step 3: Push to trigger workflows"
    echo "   git push origin main"
    echo ""
fi

if [ "$NEEDS_ADD" = false ] && [ "$NEEDS_COMMIT" = false ] && [ "$NEEDS_PUSH" = false ]; then
    echo "✅ Everything looks good!"
    echo ""
    echo "Your workflows should run on the next push."
    echo ""
    echo "To verify, visit:"
    echo "https://github.com/sandeshregmi/claims-autoencoder/actions"
else
    echo "After completing the steps above, check:"
    echo "https://github.com/sandeshregmi/claims-autoencoder/actions"
fi

echo ""
