# Troubleshooting Guide - Claims Autoencoder

## ❌ Problem: Cannot Run `python train.py --config config/example_config.yaml`

### Quick Diagnosis

Run this first to identify the issue:
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
python test_training_setup.py
```

This will test all components and show you exactly what's failing.

---

## 🔍 Common Issues and Solutions

### Issue 1: ModuleNotFoundError: No module named 'src'

**Symptoms:**
```
ModuleNotFoundError: No module named 'src'
ModuleNotFoundError: No module named 'src.config_manager'
```

**Solutions:**

**A. Run from correct directory:**
```bash
# Make sure you're in the claims-autoencoder directory
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Then run
python train.py --config config/example_config.yaml
```

**B. Fix train.py (FIXED):**
The train.py has been updated to correctly set the Python path. If you still have issues:
```bash
# Check current directory
pwd
# Should show: /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Try with explicit path
PYTHONPATH=. python train.py --config config/example_config.yaml
```

---

### Issue 2: Missing Dependencies

**Symptoms:**
```
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific packages
pip install torch>=2.0.0 pandas>=2.0.0 scikit-learn>=1.3.0

# For tree models
pip install xgboost>=2.0.0 catboost>=1.2.0
```

---

### Issue 3: Data File Not Found

**Symptoms:**
```
FileNotFoundError: data/claims_train.parquet not found
```

**Solution:**

**Check if file exists:**
```bash
ls -lh data/claims_train.parquet
```

**If file doesn't exist, you have options:**

**A. Create sample data:**
```python
# create_sample_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 10000

data = {
    'claim_amount': np.random.uniform(1000, 50000, n_samples),
    'patient_age': np.random.randint(18, 90, n_samples),
    'provider_experience_years': np.random.randint(1, 40, n_samples),
    'days_since_last_claim': np.random.randint(1, 365, n_samples),
    'num_previous_claims': np.random.randint(0, 20, n_samples),
    'average_claim_amount': np.random.uniform(1000, 40000, n_samples),
    'claim_duration_days': np.random.randint(1, 90, n_samples),
    'claim_type': np.random.choice(['routine', 'emergency', 'preventive'], n_samples),
    'provider_specialty': np.random.choice(['general', 'specialist', 'surgery'], n_samples),
    'diagnosis_code': np.random.choice(['D001', 'D002', 'D003', 'D004'], n_samples),
    'procedure_code': np.random.choice(['P001', 'P002', 'P003'], n_samples),
    'patient_gender': np.random.choice(['M', 'F'], n_samples),
    'geographic_region': np.random.choice(['Northeast', 'South', 'Midwest', 'West'], n_samples),
}

df = pd.DataFrame(data)
df.to_parquet('data/claims_train.parquet', index=False)
print(f"Created data/claims_train.parquet with {len(df)} samples")
```

Then run:
```bash
python create_sample_data.py
```

**B. Update config to use your data:**
Edit `config/example_config.yaml`:
```yaml
data:
  train_path: "path/to/your/data.parquet"  # or .csv
```

---

### Issue 4: Configuration Error

**Symptoms:**
```
KeyError: 'batch_size'
AttributeError: 'ConfigManager' object has no attribute 'training'
```

**Solution:**

**Check config file format:**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/example_config.yaml'))"
```

**Use starter config:**
```bash
cp config/starter_config.yaml config/my_config.yaml
python train.py --config config/my_config.yaml
```

---

### Issue 5: CUDA/GPU Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
RuntimeError: No CUDA GPUs are available
```

**Solution:**

**A. Use CPU:**
Edit config file:
```yaml
training:
  accelerator: "cpu"  # Change from "auto" or "gpu"
  devices: 1
```

**B. Reduce batch size:**
```yaml
training:
  batch_size: 128  # Down from 256
```

**C. Use MPS (Mac with M1/M2):**
```yaml
training:
  accelerator: "mps"
  devices: 1
```

---

### Issue 6: MLflow Errors

**Symptoms:**
```
mlflow.exceptions.MlflowException: ...
Permission denied: mlruns/...
```

**Solution:**

**A. Disable MLflow temporarily:**
```yaml
mlflow:
  enabled: false
```

**B. Clear MLflow directory:**
```bash
rm -rf mlruns
mkdir mlruns
```

**C. Use different tracking URI:**
```yaml
mlflow:
  tracking_uri: "mlruns_new"
```

---

## 🚀 Step-by-Step Training Guide

### 1. Verify Setup
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
python test_training_setup.py
```

### 2. Install Dependencies (if needed)
```bash
pip install -r requirements.txt
```

### 3. Verify Data
```bash
ls -lh data/claims_train.parquet
```

### 4. Start Training
```bash
# Basic training
python train.py --config config/example_config.yaml

# Training with CPU only
python train.py --config config/example_config.yaml --device cpu

# Quick test (reduce epochs in config first)
python train.py --config config/starter_config.yaml
```

### 5. Monitor Progress
```bash
# In another terminal, start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

---

## 🌳 Alternative: Try Tree Models First

Tree models are faster and easier to debug:

```bash
# Run tree model examples (no GPU needed, trains in seconds)
python examples_tree_fraud_detection.py

# Or test tree models
pytest tests/test_tree_models.py -v
```

---

## 📊 Minimal Working Example

If nothing works, try this minimal script:

```python
# minimal_test.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")
from src.config_manager import ConfigManager
from src.data_ingestion import DataIngestion
from src.preprocessing import ClaimsPreprocessor

print("Loading config...")
config = ConfigManager("config/example_config.yaml")

print("Loading data...")
data_ingestion = DataIngestion(config.get_config())
train_df, _, _ = data_ingestion.load_train_val_test()

print(f"Success! Loaded {len(train_df)} training samples")
print(f"Columns: {list(train_df.columns)}")
```

Run with:
```bash
python minimal_test.py
```

---

## 🔧 Environment Check

Run this to check your environment:

```bash
# Check Python version (need 3.8+)
python --version

# Check installed packages
pip list | grep -E "torch|pandas|scikit|xgboost|catboost"

# Check current directory
pwd

# Check file structure
ls -la
ls -la src/
ls -la config/
ls -la data/
```

---

## 📞 Still Having Issues?

### Debug Mode

Enable verbose logging:

```python
# Add to top of train.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Logs

```bash
# Check if logs are being created
ls -la logs/

# View recent errors
tail -n 50 logs/training.log
```

### Get Detailed Error

```bash
# Run with Python's verbose mode
python -v train.py --config config/example_config.yaml 2>&1 | tee debug.log
```

---

## ✅ Verification Checklist

Before running training, verify:

- [ ] In correct directory: `claims-autoencoder/`
- [ ] Dependencies installed: `pip list | grep torch`
- [ ] Data file exists: `ls data/claims_train.parquet`
- [ ] Config file valid: `cat config/example_config.yaml`
- [ ] Test script passes: `python test_training_setup.py`
- [ ] Python 3.8+: `python --version`

---

## 🎯 Quick Fixes

### Fix 1: Path Issues
```bash
export PYTHONPATH=/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder:$PYTHONPATH
python train.py --config config/example_config.yaml
```

### Fix 2: Permission Issues
```bash
chmod +x train.py
chmod +x test_training_setup.py
```

### Fix 3: Virtual Environment
```bash
# Create fresh environment
python -m venv venv_claims
source venv_claims/bin/activate  # On Mac/Linux
# or
venv_claims\Scripts\activate  # On Windows

pip install -r requirements.txt
python train.py --config config/example_config.yaml
```

---

## 📚 Additional Resources

- **Main README**: `README.md`
- **Setup Instructions**: `SETUP_INSTRUCTIONS.md`
- **Tree Models Guide**: `TREE_MODELS_FRAUD_DETECTION.md`
- **Test Setup Script**: `test_training_setup.py`

---

**Need help? Run the diagnostic:**
```bash
python test_training_setup.py
```

This will pinpoint exactly what's wrong! ✅
