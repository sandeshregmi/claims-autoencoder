# 🎯 Quick Start Guide - Claims Autoencoder

## Installation & Setup

### Method 1: Install as Package (Recommended)
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Install in development mode
pip install -e .

# Now you can run from anywhere:
train-autoencoder --config config/example_config.yaml
```

### Method 2: Use Wrapper Scripts (Easiest)
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py --config config/example_config.yaml

# Score claims
python score.py --config config/example_config.yaml \
    --model-path models/best_model.pth \
    --preprocessor-path models/preprocessor.pkl \
    --input-path data/claims.parquet \
    --output-path results/scored.parquet

# Hyperparameter tuning
python tune.py --config config/example_config.yaml

# Run web app
streamlit run app.py
```

### Method 3: Set PYTHONPATH
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Linux/Mac
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python src/training.py --config config/example_config.yaml

# Windows
set PYTHONPATH=%PYTHONPATH%;%CD%
python src/training.py --config config/example_config.yaml
```

## 🚀 Usage Examples

### 1. Generate Sample Data
```python
import sys
sys.path.insert(0, '/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder')

from src.data_ingestion import load_sample_data

# Generate sample claims data
df = load_sample_data(n_samples=10000)
df.to_parquet('data/claims_train.parquet', index=False)

print(f"Created sample data with {len(df)} claims")
```

### 2. Train Model
```bash
# Using wrapper script (easiest)
python train.py --config config/example_config.yaml

# Or with installed package
train-autoencoder --config config/example_config.yaml

# With custom device
python train.py --config config/example_config.yaml --device cuda
```

### 3. Evaluate Model
```python
import sys
sys.path.insert(0, '/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder')

import torch
import pandas as pd
from src.preprocessing import ClaimsPreprocessor
from src.evaluation import ModelEvaluator
from src.config_manager import ConfigManager

# Load config
config_manager = ConfigManager('config/example_config.yaml')
config = config_manager.get_config()

# Load model and preprocessor
model = torch.load('models/best_model.pth')
preprocessor = ClaimsPreprocessor.load('models/preprocessor.pkl', config)

# Load test data
test_df = pd.read_parquet('data/claims_test.parquet')
X_test = preprocessor.transform(test_df)

# Evaluate
evaluator = ModelEvaluator(config, model)
evaluator.set_anomaly_threshold(X_test)
metrics = evaluator.evaluate(X_test)

print(metrics)
```

### 4. Batch Score Claims
```bash
python score.py \
    --config config/example_config.yaml \
    --model-path models/best_model.pth \
    --preprocessor-path models/preprocessor.pkl \
    --input-path data/new_claims.parquet \
    --output-path results/scored_claims.parquet \
    --threshold 0.05
```

### 5. Run Web Dashboard
```bash
# Using wrapper
streamlit run app.py

# Or directly (with PYTHONPATH set)
streamlit run src/webapp.py
```

### 6. Monitor for Drift
```python
import sys
sys.path.insert(0, '/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder')

import pandas as pd
from src.preprocessing import ClaimsPreprocessor
from src.psi_monitoring import PSIMonitor
from src.config_manager import ConfigManager

# Load data
config_manager = ConfigManager('config/example_config.yaml')
config = config_manager.get_config()
preprocessor = ClaimsPreprocessor.load('models/preprocessor.pkl', config)

reference_df = pd.read_parquet('data/claims_train.parquet')
current_df = pd.read_parquet('data/claims_current.parquet')

# Preprocess
X_ref = preprocessor.transform(reference_df)
X_curr = preprocessor.transform(current_df)

# Check drift
feature_names = preprocessor.get_feature_names()
monitor = PSIMonitor(X_ref, feature_names=feature_names)
results = monitor.detect_drift(X_curr)

print(f"Overall PSI: {results['overall_psi']:.4f}")
print(f"Drift Status: {results['drift_status']}")
print(f"Major drift features: {results['drifted_features']['major']}")
```

## 🧪 Running Tests

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model_architecture.py -v

# Run specific test
pytest tests/test_model_architecture.py::TestClaimsAutoencoder::test_forward_pass -v
```

## 📝 Common Issues & Solutions

### Issue: ModuleNotFoundError: No module named 'src'

**Solution 1** (Easiest): Use the wrapper scripts
```bash
python train.py --config config/example_config.yaml
```

**Solution 2**: Install as package
```bash
pip install -e .
```

**Solution 3**: Set PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: FileNotFoundError: Data file not found

**Solution**: Create sample data first
```python
from src.data_ingestion import load_sample_data
df = load_sample_data(n_samples=10000)
df.to_parquet('data/claims_train.parquet')
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size in config or use CPU
```yaml
training:
  batch_size: 128  # Reduce from 256
  accelerator: "cpu"  # Use CPU instead of GPU
```

### Issue: Import errors in Streamlit

**Solution**: Use the app.py wrapper
```bash
streamlit run app.py
```

## 📦 Project Structure Quick Reference

```
claims-autoencoder/
├── train.py          # Training wrapper
├── score.py          # Scoring wrapper  
├── tune.py           # Tuning wrapper
├── app.py            # Webapp wrapper
├── setup.py          # Package installation
├── src/              # Source code
├── tests/            # Test suite
├── config/           # Configuration files
├── data/             # Data directory (create this)
├── models/           # Saved models (created automatically)
└── results/          # Output results (created automatically)
```

## 🎯 Typical Workflow

1. **Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create/Prepare Data**
   ```python
   from src.data_ingestion import load_sample_data
   df = load_sample_data(10000)
   df.to_parquet('data/claims_train.parquet')
   ```

3. **Train Model**
   ```bash
   python train.py --config config/example_config.yaml
   ```

4. **Evaluate**
   ```bash
   # Check MLflow UI
   mlflow ui
   # Navigate to http://localhost:5000
   ```

5. **Score New Data**
   ```bash
   python score.py --config config/example_config.yaml \
       --model-path models/best_model.pth \
       --preprocessor-path models/preprocessor.pkl \
       --input-path data/new_claims.parquet \
       --output-path results/scored.parquet
   ```

6. **Monitor (Optional)**
   ```bash
   streamlit run app.py
   ```

## 💡 Tips

- Always run commands from the project root directory
- Use the wrapper scripts (`train.py`, `score.py`, etc.) for easiest execution
- Check `config/example_config.yaml` and adjust for your data
- Start with small sample data to test the pipeline
- Monitor training with `mlflow ui` in a separate terminal

## 📞 Need Help?

Check the full README.md for detailed documentation of all features and modules.
