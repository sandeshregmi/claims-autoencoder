#!/bin/bash

# Clean Workflow Pipeline - Remove Unnecessary Files
# This script removes duplicate, outdated, and temporary files

set -e

BASE_DIR="/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder"
cd "$BASE_DIR"

echo "Starting cleanup of unnecessary files..."
echo "============================================"

# Create backup directory
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Backup directory created: $BACKUP_DIR"

# Function to safely remove files
safe_remove() {
    local pattern=$1
    local description=$2
    echo ""
    echo "Removing: $description"
    find . -name "$pattern" -type f -print -exec mv {} "$BACKUP_DIR/" \;
}

# Function to safely remove directories
safe_remove_dir() {
    local pattern=$1
    local description=$2
    echo ""
    echo "Removing: $description"
    find . -name "$pattern" -type d -exec mv {} "$BACKUP_DIR/" \;
}

# 1. Remove documentation files (keep README.md and QUICKSTART.md)
echo ""
echo "Step 1: Cleaning documentation files..."
safe_remove "*_COMPLETE.md" "Completion documentation"
safe_remove "*_FIX.md" "Fix documentation"
safe_remove "*_IMPLEMENTATION.md" "Implementation docs"
safe_remove "*_ARCHITECTURE.md" "Architecture docs"
safe_remove "*_CHANGELOG.md" "Changelog docs"
safe_remove "*_INTEGRATION.md" "Integration docs"
safe_remove "*_INSTRUCTIONS.md" "Instruction docs"
safe_remove "*_PLAN.md" "Plan docs"
safe_remove "*_STATUS.md" "Status docs"
safe_remove "*_SUMMARY.md" "Summary docs"
safe_remove "*_REFERENCE.md" "Reference docs (except QUICK_REFERENCE)"
safe_remove "*_TROUBLESHOOTING.md" "Troubleshooting docs"
safe_remove "SOLUTION.md" "Solution docs"

# 2. Remove duplicate application files
echo ""
echo "Step 2: Cleaning duplicate application files..."
safe_remove "app_complete.py" "Complete app versions"
safe_remove "app_enhanced.py" "Enhanced app versions"
safe_remove "src/webapp.py" "Old webapp"
safe_remove "src/webapp_enhanced_COMPLETE.py" "Complete webapp"
safe_remove "src/webapp_enhanced_backup*.py" "Webapp backups"

# 3. Remove utility scripts
echo ""
echo "Step 3: Cleaning utility scripts..."
safe_remove "add_*.py" "Add scripts"
safe_remove "apply_*.py" "Apply scripts"
safe_remove "create_*.py" "Create scripts"
safe_remove "upgrade_*.py" "Upgrade scripts"
safe_remove "fix_*.sh" "Fix shell scripts"
safe_remove "cleanup_*.sh" "Cleanup scripts"
safe_remove "force_*.py" "Force scripts"
safe_remove "quick_*.py" "Quick scripts (except necessary ones)"

# 4. Remove test files at root
echo ""
echo "Step 4: Cleaning root test files..."
safe_remove "test.py" "Test file"
safe_remove "test_*.py" "Test files"

# 5. Remove cache directories
echo ""
echo "Step 5: Cleaning cache directories..."
safe_remove_dir "__pycache__" "Python cache"
safe_remove_dir ".pytest_cache" "Pytest cache"
safe_remove ".DS_Store" "macOS files"

# 6. Remove CatBoost info
echo ""
echo "Step 6: Cleaning CatBoost info..."
safe_remove_dir "catboost_info" "CatBoost info"

# 7. Clean up old checkpoints (keep only models directory)
echo ""
echo "Step 7: Cleaning old checkpoints..."
if [ -d "checkpoints" ]; then
    echo "Archiving old checkpoints to $BACKUP_DIR/checkpoints/"
    mv checkpoints "$BACKUP_DIR/" || true
fi

# 8. Clean up databricks (if not used)
echo ""
echo "Step 8: Cleaning Databricks files (optional)..."
read -p "Remove Databricks files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    safe_remove_dir ".databricks" "Databricks directory"
    safe_remove "databricks.yml" "Databricks config"
    safe_remove "databricks_*.py" "Databricks scripts"
    safe_remove "databricks_*.md" "Databricks docs"
    safe_remove "deploy_databricks.sh" "Databricks deployment"
fi

# 9. Clean up MLruns (if not needed)
echo ""
echo "Step 9: Cleaning MLflow runs..."
read -p "Archive MLflow runs? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "mlruns" ]; then
        echo "Archiving mlruns to $BACKUP_DIR/mlruns/"
        mv mlruns "$BACKUP_DIR/" || true
    fi
fi

# 10. Clean up old results
echo ""
echo "Step 10: Cleaning old results..."
if [ -d "results/model_comparison" ]; then
    echo "Archiving old model comparison logs..."
    find results/model_comparison -name "*.log" -type f -exec mv {} "$BACKUP_DIR/" \;
fi

echo ""
echo "============================================"
echo "Cleanup complete!"
echo ""
echo "Backup location: $BASE_DIR/$BACKUP_DIR"
echo ""
echo "Essential files kept:"
echo "  - src/webapp_enhanced.py (main app)"
echo "  - src/tree_models.py"
echo "  - src/preprocessing.py"
echo "  - src/config_manager.py"
echo "  - src/data_ingestion.py"
echo "  - src/psi_monitoring.py"
echo "  - src/fairness_analysis.py"
echo "  - shap_explainer.py"
echo "  - config/starter_config.yaml"
echo "  - data/claims_train.parquet"
echo "  - requirements.txt"
echo "  - README.md"
echo "  - QUICKSTART.md"
echo ""
echo "You can delete the backup directory if everything works:"
echo "  rm -rf $BACKUP_DIR"
