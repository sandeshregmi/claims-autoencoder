#!/usr/bin/env python3
"""
Quick verification script to check if modularization was successful
Run this AFTER: pip install -e .
"""

import sys
from pathlib import Path

def check_structure():
    """Check if new structure exists"""
    print("1. Checking package structure...")
    base = Path("src/claims_fraud")
    
    required_dirs = [
        "core", "data", "analysis", "ml", "ui", "utils", "cli"
    ]
    
    for dir_name in required_dirs:
        dir_path = base / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ - MISSING!")
            return False
    
    print("   ✅ All directories present\n")
    return True

def check_imports():
    """Check if package can be imported"""
    print("2. Checking Python imports...")
    
    try:
        import claims_fraud
        print(f"   ✅ claims_fraud imported - version {claims_fraud.__version__}")
    except ImportError as e:
        print(f"   ❌ Cannot import claims_fraud: {e}")
        print("   → Run: pip install -e .")
        return False
    
    # Check main classes
    try:
        from claims_fraud import FraudDetector, TreeModel
        print(f"   ✅ Main classes importable")
    except ImportError as e:
        print(f"   ❌ Cannot import main classes: {e}")
        return False
    
    print()
    return True

def check_cli():
    """Check if CLI is available"""
    print("3. Checking CLI availability...")
    
    import subprocess
    try:
        result = subprocess.run(
            ["claims-fraud", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"   ✅ CLI available: {result.stdout.strip()}")
        else:
            print(f"   ⚠️  CLI found but returned error")
            return False
    except FileNotFoundError:
        print("   ❌ CLI not found")
        print("   → Run: pip install -e . --force-reinstall")
        return False
    except Exception as e:
        print(f"   ⚠️  Error checking CLI: {e}")
        return False
    
    print()
    return True

def check_files():
    """Check if key files exist"""
    print("4. Checking key files...")
    
    files = [
        "pyproject.toml",
        "README.md",
        "MANIFEST.in",
        "src/claims_fraud/__init__.py",
        "src/claims_fraud/__version__.py",
    ]
    
    all_exist = True
    for file in files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING!")
            all_exist = False
    
    print()
    return all_exist

def main():
    print("=" * 60)
    print("Claims Fraud Detection - Installation Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Structure", check_structure),
        ("Imports", check_imports),
        ("CLI", check_cli),
        ("Files", check_files),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ Error in {name} check: {e}\n")
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")
    
    print()
    print(f"Score: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your package is ready to use.")
        print("\nNext steps:")
        print("  1. Launch dashboard: claims-fraud serve")
        print("  2. Try CLI: claims-fraud --help")
        print("  3. Test API: python examples/quickstart.py")
    else:
        print("\n⚠️  Some checks failed. Please:")
        print("  1. Run: pip install -e .")
        print("  2. Check NEXT_STEPS_AFTER_MODULARIZATION.md")
        print("  3. Run this script again")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
