#!/usr/bin/env python3
"""
Fix import paths in migrated files
Run this after modularization to fix remaining import issues
"""

import re
from pathlib import Path

def fix_imports_in_file(file_path: Path):
    """Fix import statements in a single file"""
    content = file_path.read_text()
    original = content
    
    # Fix patterns
    patterns = [
        # Fix: from claims_fraud.config_manager → from claims_fraud.config.manager
        (r'from claims_fraud\.config_manager import', 'from claims_fraud.config.manager import'),
        (r'import claims_fraud\.config_manager', 'import claims_fraud.config.manager'),
        
        # Fix: from claims_fraud.preprocessing → from claims_fraud.data.preprocessing  
        (r'from claims_fraud\.preprocessing import', 'from claims_fraud.data.preprocessing import'),
        (r'import claims_fraud\.preprocessing', 'import claims_fraud.data.preprocessing'),
        
        # Fix: from claims_fraud.data_ingestion → from claims_fraud.data.ingestion
        (r'from claims_fraud\.data_ingestion import', 'from claims_fraud.data.ingestion import'),
        (r'import claims_fraud\.data_ingestion', 'import claims_fraud.data.ingestion'),
        
        # Fix: from claims_fraud.model_architecture → from claims_fraud.core.base
        (r'from claims_fraud\.model_architecture import', 'from claims_fraud.core.base import'),
        (r'import claims_fraud\.model_architecture', 'import claims_fraud.core.base'),
        
        # Fix: from claims_fraud.tree_models → from claims_fraud.core.tree_models
        (r'from claims_fraud\.tree_models import', 'from claims_fraud.core.tree_models import'),
        (r'import claims_fraud\.tree_models', 'import claims_fraud.core.tree_models'),
        
        # Fix: from claims_fraud.training → from claims_fraud.ml.training
        (r'from claims_fraud\.training import', 'from claims_fraud.ml.training import'),
        (r'import claims_fraud\.training', 'import claims_fraud.ml.training'),
        
        # Fix: from claims_fraud.evaluation → from claims_fraud.analysis.evaluation
        (r'from claims_fraud\.evaluation import', 'from claims_fraud.analysis.evaluation import'),
        (r'import claims_fraud\.evaluation', 'import claims_fraud.analysis.evaluation'),
        
        # Fix: from claims_fraud.fairness_analysis → from claims_fraud.analysis.fairness
        (r'from claims_fraud\.fairness_analysis import', 'from claims_fraud.analysis.fairness import'),
        (r'import claims_fraud\.fairness_analysis', 'import claims_fraud.analysis.fairness'),
        
        # Fix: from claims_fraud.psi_monitoring → from claims_fraud.analysis.monitoring
        (r'from claims_fraud\.psi_monitoring import', 'from claims_fraud.analysis.monitoring import'),
        (r'import claims_fraud\.psi_monitoring', 'import claims_fraud.analysis.monitoring'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        file_path.write_text(content)
        return True
    return False

def main():
    """Fix all imports in the claims_fraud package"""
    print("=" * 60)
    print("Fixing Import Paths")
    print("=" * 60)
    print()
    
    base_path = Path("src/claims_fraud")
    
    if not base_path.exists():
        print("❌ src/claims_fraud not found!")
        return 1
    
    # Find all Python files
    py_files = list(base_path.rglob("*.py"))
    
    print(f"Found {len(py_files)} Python files")
    print()
    
    fixed_count = 0
    for file_path in py_files:
        relative_path = file_path.relative_to(base_path)
        if fix_imports_in_file(file_path):
            print(f"✅ Fixed: {relative_path}")
            fixed_count += 1
    
    print()
    print("=" * 60)
    if fixed_count > 0:
        print(f"✅ Fixed {fixed_count} files")
        print()
        print("Next steps:")
        print("  1. Reinstall: pip install -e . --force-reinstall")
        print("  2. Verify: python verify_installation.py")
    else:
        print("✅ No import issues found!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
