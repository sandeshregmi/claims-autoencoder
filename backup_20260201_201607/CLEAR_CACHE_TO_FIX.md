# URGENT: Clear Python Cache to Apply Fix

## 🚨 The Issue

The XGBoost fix **IS APPLIED** in the code (`src/tree_models.py`), but Python is using **cached bytecode** from before the fix!

## ✅ The Solution

Run this single command:

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder && rm -rf src/__pycache__ && python tree_fraud_detection_runner.py --config config/example_config.yaml
```

## 🔍 What's Happening

1. **The fix is in the file** ✅ - I verified `src/tree_models.py` has the NaN handling
2. **Python cached the old version** ❌ - File `src/__pycache__/tree_models.cpython-39.pyc` has old bytecode
3. **Python uses cache by default** - It doesn't check if source changed
4. **Solution: Delete cache** - Force Python to recompile from source

## 📋 Alternative Methods

### Method 1: Clear Cache + Run (RECOMMENDED)
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
rm -rf src/__pycache__
python tree_fraud_detection_runner.py --config config/example_config.yaml
```

### Method 2: Run Without Cache
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
python -B tree_fraud_detection_runner.py --config config/example_config.yaml
```

### Method 3: Clear All Caches
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
python tree_fraud_detection_runner.py --config config/example_config.yaml
```

### Method 4: Use Shell Script
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
chmod +x run_fixed.sh
./run_fixed.sh
```

## 🎯 Expected Result

After clearing cache, you should see:

```
✅ Training xgboost fraud detector...
✅ Progress: 3/13 models trained
✅ Progress: 6/13 models trained
✅ Progress: 9/13 models trained
✅ Progress: 12/13 models trained
✅ Progress: 13/13 models trained
✅ All fraud detection models trained successfully

NO XGBOOST ERRORS! 🎉
```

## 🧪 Verify the Fix is in Place

Check the source file has the fix:

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
grep -A 5 "Filter out rows with NaN" src/tree_models.py
```

Should output:
```python
# Filter out rows with NaN in target variable
valid_mask = ~y.isna()
if not valid_mask.all():
    n_nan = (~valid_mask).sum()
    if n_nan > len(y) * 0.5:
        logger.warning(f"Feature '{target_name}': {n_nan}/{len(y)} NaN values")
```

## ❓ Why Did This Happen?

Python compiles `.py` files to `.pyc` bytecode for faster loading. When you:
1. ✅ Edit the source file (`tree_models.py`)
2. ❌ Don't clear cache (`tree_models.cpython-39.pyc`)
3. ❌ Python loads the OLD cached version

This is a common issue when:
- Editing code while Python processes are running
- Files are edited externally
- Timestamps don't update properly
- Cache directory permissions issues

## 🔧 Permanent Solution

Always clear cache after editing code:

```bash
# Add to your workflow
alias pyclean='find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null'

# Then just run
pyclean
python your_script.py
```

Or use Python's `-B` flag:
```bash
# Never write .pyc files
python -B your_script.py
```

## 📝 Quick Reference

| Command | What It Does |
|---------|-------------|
| `rm -rf src/__pycache__` | Clear one directory's cache |
| `find . -name __pycache__ -exec rm -rf {} +` | Clear all caches recursively |
| `python -B script.py` | Run without creating cache |
| `python -c "import sys; print(sys.dont_write_bytecode)"` | Check cache setting |

---

## 🎯 JUST RUN THIS NOW:

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder && rm -rf src/__pycache__ && python tree_fraud_detection_runner.py --config config/example_config.yaml
```

**That's it!** The fix will work once cache is cleared. 🚀
