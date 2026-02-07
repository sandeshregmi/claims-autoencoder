# 🎯 QUICK FIX GUIDE - Training Issues

## ❌ Problem: Cannot run `python train.py --config config/example_config.yaml`

---

## ✅ SOLUTION 1: Run Diagnostic (Recommended)

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
python test_training_setup.py
```

**This will tell you exactly what's wrong!**

---

## ✅ SOLUTION 2: Most Common Fixes

### Fix #1: Wrong Directory
```bash
# Make sure you're HERE:
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Then run:
python train.py --config config/example_config.yaml
```

### Fix #2: Missing Dependencies
```bash
pip install -r requirements.txt
```

### Fix #3: Path Issues (train.py has been fixed)
```bash
# Use explicit path
PYTHONPATH=. python train.py --config config/example_config.yaml
```

---

## ✅ SOLUTION 3: Try Tree Models Instead (Easier!)

Tree models are faster and easier to get working:

```bash
# Install tree model dependencies
pip install xgboost>=2.0.0 catboost>=1.2.0

# Run examples (works immediately, no setup needed!)
python examples_tree_fraud_detection.py

# Or run tests
pytest tests/test_tree_models.py -v
```

This will show you the tree models working right away! 🚀

---

## 📋 Complete Troubleshooting

See **`TROUBLESHOOTING.md`** for detailed solutions to all common issues.

---

## 🎯 Files I Created to Help You

### 1. Diagnostic Tool
**`test_training_setup.py`** - Identifies exactly what's wrong
```bash
python test_training_setup.py
```

### 2. Troubleshooting Guide
**`TROUBLESHOOTING.md`** - Complete solutions for all issues

### 3. Tree Models (Alternative)
If neural training is complex, use tree models:
- **`examples_tree_fraud_detection.py`** - Working examples
- **`TREE_MODELS_FRAUD_DETECTION.md`** - Complete guide

---

## 🚀 Quick Start (3 Steps)

```bash
# 1. Go to correct directory
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# 2. Run diagnostic
python test_training_setup.py

# 3. Follow the output recommendations
```

---

## 💡 What Was Fixed

I fixed the `train.py` file - it was trying to add the wrong directory to the Python path. It's now corrected.

**Before:**
```python
project_root = Path(__file__).parent.parent  # WRONG
```

**After:**
```python
project_root = Path(__file__).parent  # CORRECT
```

---

## 🎉 Summary

You now have:
- ✅ Fixed `train.py` 
- ✅ Diagnostic tool (`test_training_setup.py`)
- ✅ Complete troubleshooting guide (`TROUBLESHOOTING.md`)
- ✅ Working tree model examples (alternative approach)

**Next step**: Run `python test_training_setup.py` to see what (if anything) is still wrong!
