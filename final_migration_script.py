#!/usr/bin/env python3
"""
Final Complete Migration Script
Migrates all remaining files from backup to new package structure
"""

import re
from pathlib import Path
import shutil

# Configuration
BACKUP_DIR = Path("_modularization_backup_20260204_173929/src")
NEW_PKG = Path("src/claims_fraud")
ORIGINALS_DIR = Path("src/_originals")

# Migration mapping with import replacements
MIGRATIONS = [
    {
        "name": "data_ingestion.py",
        "source": BACKUP_DIR / "data_ingestion.py",
        "dest": NEW_PKG / "data/ingestion.py",
        "imports": [
            (r"from src\.config_manager import", "from ..config.manager import"),
        ]
    },
    {
        "name": "preprocessing.py",
        "source": BACKUP_DIR / "preprocessing.py",
        "dest": NEW_PKG / "data/preprocessing.py",
        "imports": [
            (r"from src\.config_manager import", "from ..config.manager import"),
        ]
    },
    {
        "name": "tree_models.py",
        "source": BACKUP_DIR / "tree_models.py",
        "dest": NEW_PKG / "core/tree_models.py",
        "imports": [
            (r"from src\.config_manager import", "from ..config.manager import"),
            (r"from src\.preprocessing import", "from ..data.preprocessing import"),
        ]
    },
    {
        "name": "fairness_analysis.py",
        "source": BACKUP_DIR / "fairness_analysis.py",
        "dest": NEW_PKG / "analysis/fairness.py",
        "imports": [
            (r"from src\.config_manager import", "from ..config.manager import"),
            (r"from src\.tree_models import", "from ..core.tree_models import"),
            (r"from src\.preprocessing import", "from ..data.preprocessing import"),
        ]
    },
    {
        "name": "psi_monitoring.py",
        "source": BACKUP_DIR / "psi_monitoring.py",
        "dest": NEW_PKG / "analysis/monitoring.py",
        "imports": [
            (r"from src\.config_manager import", "from ..config.manager import"),
            (r"from src\.preprocessing import", "from ..data.preprocessing import"),
        ]
    },
    {
        "name": "evaluation.py",
        "source": BACKUP_DIR / "evaluation.py",
        "dest": NEW_PKG / "analysis/evaluation.py",
        "imports": [
            (r"from src\.config_manager import", "from ..config.manager import"),
            (r"from src\.tree_models import", "from ..core.tree_models import"),
        ]
    },
    {
        "name": "training.py",
        "source": BACKUP_DIR / "training.py",
        "dest": NEW_PKG / "ml/training.py",
        "imports": [
            (r"from src\.config_manager import", "from ..config.manager import"),
            (r"from src\.tree_models import", "from ..core.tree_models import"),
            (r"from src\.preprocessing import", "from ..data.preprocessing import"),
            (r"from src\.data_ingestion import", "from ..data.ingestion import"),
        ]
    },
    {
        "name": "hyperparameter_tuning.py",
        "source": BACKUP_DIR / "hyperparameter_tuning.py",
        "dest": NEW_PKG / "ml/tuning.py",
        "imports": [
            (r"from src\.config_manager import", "from ..config.manager import"),
            (r"from src\.training import", "from .training import"),
            (r"from src\.tree_models import", "from ..core.tree_models import"),
        ]
    },
    {
        "name": "model_registry.py",
        "source": BACKUP_DIR / "model_registry.py",
        "dest": NEW_PKG / "ml/registry.py",
        "imports": [
            (r"from src\.config_manager import", "from ..config.manager import"),
        ]
    },
    {
        "name": "batch_scoring.py",
        "source": BACKUP_DIR / "batch_scoring.py",
        "dest": NEW_PKG / "core/scoring.py",
        "imports": [
            (r"from src\.config_manager import", "from ..config.manager import"),
            (r"from src\.preprocessing import", "from ..data.preprocessing import"),
            (r"from src\.tree_models import", "from .tree_models import"),
        ]
    },
    {
        "name": "webapp_enhanced.py",
        "source": BACKUP_DIR / "webapp_enhanced.py",
        "dest": NEW_PKG / "ui/app.py",
        "imports": [
            (r"from src\.config_manager import", "from ..config.manager import"),
            (r"from src\.data_ingestion import", "from ..data.ingestion import"),
            (r"from src\.tree_models import", "from ..core.tree_models import"),
            (r"from src\.psi_monitoring import", "from ..analysis.monitoring import"),
            (r"from src\.fairness_analysis import", "from ..analysis.fairness import"),
        ]
    },
]


def migrate_file(migration):
    """Migrate a single file"""
    print(f"\n{'='*70}")
    print(f"Migrating: {migration['name']}")
    print('='*70)
    
    source = migration['source']
    dest = migration['dest']
    
    if not source.exists():
        print(f"❌ Source not found: {source}")
        return False
    
    try:
        # Read content
        print(f"📖 Reading: {source}")
        content = source.read_text()
        
        # Update imports
        for pattern, replacement in migration['imports']:
            before_count = len(re.findall(pattern, content))
            content = re.sub(pattern, replacement, content)
            if before_count > 0:
                print(f"  ✓ Updated {before_count} import(s): {pattern[:40]}...")
        
        # Ensure destination directory exists
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to new location
        dest.write_text(content)
        print(f"✅ Written to: {dest}")
        
        # Copy to originals backup
        backup_path = ORIGINALS_DIR / migration['name']
        ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup_path)
        print(f"💾 Backup saved: {backup_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main migration function"""
    print("\n" + "="*70)
    print("CLAIMS FRAUD DETECTION - FINAL MIGRATION")
    print("="*70)
    print()
    print(f"Source: {BACKUP_DIR}")
    print(f"Destination: {NEW_PKG}")
    print(f"Backups: {ORIGINALS_DIR}")
    print()
    
    # Migrate all files
    success_count = 0
    failed_count = 0
    
    for migration in MIGRATIONS:
        if migrate_file(migration):
            success_count += 1
        else:
            failed_count += 1
    
    # Print summary
    print("\n" + "="*70)
    print("MIGRATION SUMMARY")
    print("="*70)
    print(f"✅ Successful: {success_count}/{len(MIGRATIONS)}")
    print(f"❌ Failed: {failed_count}/{len(MIGRATIONS)}")
    print()
    
    if success_count == len(MIGRATIONS):
        print("🎉 ALL FILES MIGRATED SUCCESSFULLY!")
        print()
        print("Next steps:")
        print("  1. Install package:")
        print("     pip install -e . --break-system-packages")
        print()
        print("  2. Test import:")
        print("     python -c 'import claims_fraud; print(claims_fraud.__version__)'")
        print()
        print("  3. Launch dashboard:")
        print("     claims-fraud serve")
        print()
        print("  OR run old dashboard while testing:")
        print("     streamlit run _modularization_backup_*/src/webapp_enhanced.py")
    else:
        print("⚠️  Some files failed. Check errors above.")
        print("You can retry failed files individually.")
    
    print("="*70)
    
    return success_count, failed_count


if __name__ == "__main__":
    success, failed = main()
    exit(0 if failed == 0 else 1)
