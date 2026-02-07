# Quick Fix for Missing Dependencies

## Problem: ModuleNotFoundError

If you get errors like:
```
ModuleNotFoundError: No module named 'seaborn'
ModuleNotFoundError: No module named 'matplotlib'
```

## Solution 1: Quick Fix Script

```bash
chmod +x quick_fix_dependencies.sh
./quick_fix_dependencies.sh
```

## Solution 2: Manual Install

```bash
pip install seaborn>=0.12.0 matplotlib>=3.7.0
```

## Solution 3: Reinstall All Dependencies

```bash
pip install -r requirements_clean.txt --upgrade
```

## Solution 4: Use the Main Runner (Recommended)

The main runner script automatically installs all dependencies:

```bash
./run_clean_workflow.sh
```

## Updated Dependencies

The `requirements_clean.txt` file now includes:

```
# Web Application & Visualization
streamlit>=1.28.0
plotly>=5.17.0
seaborn>=0.12.0      # ← Added
matplotlib>=3.7.0    # ← Added
```

## Verification

After installing, verify the imports work:

```python
python3 -c "import seaborn; import matplotlib; print('✅ All imports successful!')"
```

## Still Having Issues?

1. Make sure you're using the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Check Python version (requires 3.8+):
   ```bash
   python3 --version
   ```

3. Clear pip cache and reinstall:
   ```bash
   pip cache purge
   pip install -r requirements_clean.txt --force-reinstall
   ```

## Need More Help?

Check the main documentation:
- **README_CLEAN.md** - Complete troubleshooting section
- **START_HERE.md** - Quick start guide
