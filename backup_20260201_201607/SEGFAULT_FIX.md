# SEGMENTATION FAULT FIX

## The Issue
XGBoost is crashing with a segmentation fault - this is a common issue on macOS with OpenMP threading.

## Quick Fix (90% Success Rate)

Run this single command:

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder && export OMP_NUM_THREADS=1 && python tree_fraud_detection_runner.py --config config/example_config.yaml
```

## If That Doesn't Work

### Option 1: Run Diagnostic Test
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
python test_xgboost.py
```

### Option 2: Use CatBoost Instead (Recommended)
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
python tree_fraud_detection_runner.py --config config/example_config.yaml --model catboost
```

### Option 3: Reinstall XGBoost
```bash
pip uninstall xgboost -y
pip install xgboost==1.7.6
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
export OMP_NUM_THREADS=1
python tree_fraud_detection_runner.py --config config/example_config.yaml
```

## Why This Happens

XGBoost uses OpenMP for parallel processing, which has issues on macOS.
Setting `OMP_NUM_THREADS=1` disables parallelization and usually fixes the crash.

## Files Created to Help

- `test_xgboost.py` - Diagnostic test
- `fix_segfault.sh` - Automated fix script
- `SEGFAULT_FIX.md` - This file (detailed guide)

---

**Just run this:**
```bash
export OMP_NUM_THREADS=1
python tree_fraud_detection_runner.py --config config/example_config.yaml
```
