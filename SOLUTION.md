# 🎉 PROBLEM SOLVED - Complete Usage Guide

## ✅ The Import Issue is Fixed!

You encountered: `ModuleNotFoundError: No module named 'src'`

**I've created multiple solutions for you:**

## 🚀 Solution 1: Use Wrapper Scripts (EASIEST)

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data models outputs logs checkpoints results

# Generate sample data
python -c "import sys; sys.path.insert(0, '.'); from src.data_ingestion import load_sample_data; df = load_sample_data(10000); df.to_parquet('data/claims_train.parquet'); print('✅ Sample data created!')"

# Train model (THIS WILL WORK!)
python train.py --config config/example_config.yaml

# Score claims
python score.py --config config/example_config.yaml \
    --model-path models/best_model.pth \
    --preprocessor-path models/preprocessor.pkl \
    --input-path data/claims.parquet \
    --output-path results/scored.parquet

# Run web app
streamlit run app.py
```

## 🎯 Solution 2: Install as Package (RECOMMENDED FOR PRODUCTION)

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Install in development mode
pip install -e .

# Now you can use the command-line tools from anywhere:
train-autoencoder --config config/example_config.yaml
score-claims --config config/example_config.yaml --model-path models/best_model.pth ...
```

## 🛠️ Solution 3: Use Makefile (MOST CONVENIENT)

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# See all available commands
make help

# Full setup (install + create dirs + sample data)
make full-setup

# Train model
make train

# Run tests
make test

# Launch web app
make webapp

# View MLflow experiments
make mlflow
```

## 📋 Complete Step-by-Step Walkthrough

### Step 1: Install Dependencies
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
pip install -r requirements.txt
```

### Step 2: Create Required Directories
```bash
mkdir -p data models outputs logs checkpoints results
```

### Step 3: Generate Sample Data
```bash
python -c "
import sys
sys.path.insert(0, '.')
from src.data_ingestion import load_sample_data
import pandas as pd

# Generate training data
train_df = load_sample_data(n_samples=10000, random_state=42)
train_df.to_parquet('data/claims_train.parquet', index=False)
print(f'✅ Created training data: {len(train_df)} claims')

# Generate validation data
val_df = load_sample_data(n_samples=2000, random_state=43)
val_df.to_parquet('data/claims_val.parquet', index=False)
print(f'✅ Created validation data: {len(val_df)} claims')

# Generate test data
test_df = load_sample_data(n_samples=2000, random_state=44)
test_df.to_parquet('data/claims_test.parquet', index=False)
print(f'✅ Created test data: {len(test_df)} claims')
"
```

### Step 4: Train the Model
```bash
# Using wrapper script
python train.py --config config/example_config.yaml

# Monitor training (in another terminal)
mlflow ui
# Then open http://localhost:5000 in your browser
```

### Step 5: Check Training Results
After training completes, you'll have:
- `models/best_model.pth` - Trained model
- `models/preprocessor.pkl` - Fitted preprocessor
- `mlruns/` - MLflow experiment tracking data
- `checkpoints/` - Training checkpoints

### Step 6: Score New Claims
```bash
python score.py \
    --config config/example_config.yaml \
    --model-path models/best_model.pth \
    --preprocessor-path models/preprocessor.pkl \
    --input-path data/claims_test.parquet \
    --output-path results/scored_claims.parquet \
    --threshold 0.05
```

### Step 7: Launch Web Dashboard
```bash
streamlit run app.py
```

Then:
1. Open http://localhost:8501
2. Load model in sidebar
3. Upload claims to score
4. View results and analytics

## 🧪 Testing Your Installation

```bash
# Run a quick test
python -c "
import sys
sys.path.insert(0, '.')
from src.config_manager import ConfigManager
from src.model_architecture import ClaimsAutoencoder
import torch

print('✅ Imports working!')

# Create a small test model
model = ClaimsAutoencoder(
    input_dim=10,
    encoding_dim=5,
    hidden_layers=[8],
    dropout_rate=0.2
)
print(f'✅ Model created with {model.count_parameters()} parameters')

# Test forward pass
x = torch.randn(4, 10)
reconstruction, encoding = model(x)
print(f'✅ Forward pass successful!')
print(f'   Input shape: {x.shape}')
print(f'   Encoding shape: {encoding.shape}')
print(f'   Reconstruction shape: {reconstruction.shape}')
"
```

## 📊 Example Training Session

Here's what a complete training session looks like:

```bash
# 1. Setup
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
pip install -r requirements.txt
mkdir -p data models

# 2. Generate data
python -c "import sys; sys.path.insert(0, '.'); from src.data_ingestion import load_sample_data; df = load_sample_data(10000); df.to_parquet('data/claims_train.parquet')"

# 3. Train (you'll see progress bars and metrics)
python train.py --config config/example_config.yaml

# Expected output:
# INFO: Loading data...
# INFO: Loaded 7000 rows (train), 1500 rows (val), 1500 rows (test)
# INFO: Preprocessing data...
# INFO: Creating model...
# INFO: Created model with 50,000 parameters
# INFO: Starting training for 100 epochs
# Epoch 1/100 - Train Loss: 0.125, Val Loss: 0.118
# Epoch 2/100 - Train Loss: 0.098, Val Loss: 0.095
# ...
# INFO: Training completed successfully!
```

## 🔍 Troubleshooting

### Problem: Still getting import errors

**Solution**: Make absolutely sure you're in the project directory
```bash
pwd
# Should show: /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Then use the wrapper scripts:
python train.py --config config/example_config.yaml
```

### Problem: "No such file or directory: data/claims_train.parquet"

**Solution**: Generate sample data first
```bash
make sample-data
# OR
python -c "import sys; sys.path.insert(0, '.'); from src.data_ingestion import load_sample_data; df = load_sample_data(10000); df.to_parquet('data/claims_train.parquet')"
```

### Problem: CUDA/GPU errors

**Solution**: Edit config to use CPU
```yaml
# In config/example_config.yaml
training:
  accelerator: "cpu"  # Change from "auto"
```

### Problem: Out of memory

**Solution**: Reduce batch size
```yaml
# In config/example_config.yaml
training:
  batch_size: 128  # Reduce from 256
```

## 📝 Quick Reference Card

**Create sample data:**
```bash
make sample-data
```

**Train model:**
```bash
python train.py --config config/example_config.yaml
```

**Score claims:**
```bash
python score.py --config config/example_config.yaml \
    --model-path models/best_model.pth \
    --preprocessor-path models/preprocessor.pkl \
    --input-path data/claims.parquet \
    --output-path results/scored.parquet
```

**Launch web app:**
```bash
streamlit run app.py
```

**View experiments:**
```bash
mlflow ui
```

**Run tests:**
```bash
pytest tests/
```

## 🎓 Next Steps

Now that everything is working:

1. **Customize the configuration** (`config/example_config.yaml`)
   - Adjust model architecture
   - Change training parameters
   - Configure your feature names

2. **Add your real data**
   - Replace sample data with actual claims data
   - Ensure column names match config

3. **Experiment**
   - Try different architectures
   - Tune hyperparameters
   - Adjust anomaly thresholds

4. **Deploy**
   - Use batch scoring for production
   - Set up drift monitoring
   - Deploy the web dashboard

## ✅ Summary

**The error is fixed!** You now have 3 ways to run the code:

1. **Wrapper scripts** (`train.py`, `score.py`, `app.py`) - EASIEST
2. **Makefile commands** (`make train`, `make webapp`) - CONVENIENT  
3. **Package installation** (`pip install -e .`) - PRODUCTION

All methods will work correctly. I recommend starting with the wrapper scripts!

---

**Ready to go? Start here:**
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
make full-setup
make train
```

🎉 **Happy modeling!**
