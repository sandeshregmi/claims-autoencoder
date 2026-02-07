# Import Error Fix - ModuleNotFoundError: No module named 'src'

## Problem

```
ModuleNotFoundError: No module named 'src'
Traceback:
File "/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_enhanced.py", line 20
    from src.config_manager import ConfigManager
```

## Root Cause

The webapp uses absolute imports like `from src.config_manager import ConfigManager`, which requires the project root to be in Python's module search path (PYTHONPATH).

## Solutions

### Solution 1: Use Updated Run Script (Recommended)

The `run_clean_workflow.sh` script has been updated to automatically set PYTHONPATH:

```bash
chmod +x run_clean_workflow.sh
./run_clean_workflow.sh
```

### Solution 2: Use Direct Run Script

A simpler script that just sets the path and runs:

```bash
chmod +x run_app_direct.sh
./run_app_direct.sh
```

### Solution 3: Manual PYTHONPATH Setup

Set the Python path manually before running:

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
export PYTHONPATH="${PWD}:${PYTHONPATH}"
streamlit run src/webapp_enhanced.py
```

### Solution 4: Run from Project Root

Always run Streamlit from the project root directory:

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
# Then activate virtual environment if you have one
source venv/bin/activate
# Set PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"
# Run the app
streamlit run src/webapp_enhanced.py
```

## What Was Fixed

### Updated Files

1. **run_clean_workflow.sh** - Added PYTHONPATH configuration
2. **run_app_direct.sh** - New simple runner script
3. **IMPORT_ERROR_FIX.md** - This troubleshooting guide

### Changes Made

```bash
# Added to run_clean_workflow.sh:
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
```

This ensures Python can find the `src` module when importing.

## Verification

After using one of the solutions above, you should see:

```
✓ Python path configured

Starting web application...

Access the dashboard at: http://localhost:8501

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

## Quick Reference

### Correct Way to Run

```bash
# From project root
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Option 1: Use the main script
./run_clean_workflow.sh

# Option 2: Use the direct script
./run_app_direct.sh

# Option 3: Manual
export PYTHONPATH="${PWD}:${PYTHONPATH}"
streamlit run src/webapp_enhanced.py
```

### Incorrect Ways (Will Fail)

```bash
# ❌ Running from src directory
cd src
streamlit run webapp_enhanced.py

# ❌ Running without PYTHONPATH
streamlit run src/webapp_enhanced.py
```

## Understanding the Issue

### File Structure
```
claims-autoencoder/          ← Project root (must be in PYTHONPATH)
├── src/
│   ├── webapp_enhanced.py   ← Uses: from src.config_manager import...
│   ├── config_manager.py
│   └── ...
└── ...
```

### Import Resolution

When Python sees `from src.config_manager import ConfigManager`, it looks for:
1. A directory called `src` in PYTHONPATH
2. A file called `config_manager.py` inside that directory

Without setting PYTHONPATH to include the project root, Python can't find the `src` module.

## Still Having Issues?

### Check Your Current Directory
```bash
pwd
# Should output: /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
```

### Verify PYTHONPATH
```bash
echo $PYTHONPATH
# Should include: /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
```

### Test Imports
```bash
python3 -c "from src.config_manager import ConfigManager; print('✅ Import successful!')"
```

### Check Virtual Environment
```bash
which python3
# If using venv, should point to: .../claims-autoencoder/venv/bin/python3
```

## Need More Help?

1. Ensure you're in the project root directory
2. Use one of the provided run scripts
3. Check that all files exist in the correct locations
4. Verify virtual environment is activated

---

**Issue resolved! Use `./run_clean_workflow.sh` to start the application.** 🚀
