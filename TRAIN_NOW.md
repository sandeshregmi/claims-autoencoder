# 🎯 FIXED: Ready to Train!

## ✅ What Was Fixed

The configuration has been updated to handle single training files with automatic splitting.

**Changed in `config/example_config.yaml`:**
```yaml
data:
  train_path: "data/claims_train.parquet"
  val_path: null  # Now auto-splits from train_path
  test_path: null  # Now auto-splits from train_path
```

## 🚀 Now You Can Train!

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Option 1: Use the updated config
python train.py --config config/example_config.yaml

# Option 2: Use the starter config (cleaner)
python train.py --config config/starter_config.yaml
```

## 🧪 Test Your Setup

```bash
# Run the setup test
python test_setup.py

# This will check:
# ✅ Imports working
# ✅ Configuration valid
# ✅ Data files present
# ✅ Directories exist
# ✅ Model creation
# ✅ Preprocessing pipeline
```

## 📊 What Happens During Training

With a single data file, the system will:
1. Load `data/claims_train.parquet` (10,000 rows)
2. **Auto-split** into:
   - Training: 7,000 rows (70%)
   - Validation: 1,500 rows (15%)
   - Test: 1,500 rows (15%)
3. Train the model
4. Save to `models/best_model.pth`

## 🎓 Expected Output

```
INFO: Loading data...
INFO: Loaded 10000 rows and 13 columns
INFO: Splitting training data into train/val/test sets
INFO: Train: 7000, Val: 1500, Test: 1500
INFO: Preprocessing data...
INFO: Creating model...
INFO: Created model with 50,432 parameters
INFO: Starting training for 50 epochs
Epoch 1/50: 100%|████████| 28/28 [00:02<00:00, 10.5it/s]
INFO: Epoch 1/50 - Train Loss: 0.124523, Val Loss: 0.118234
...
```

## 📁 Files Created After Training

```
models/
├── best_model.pth          # Trained model
├── preprocessor.pkl        # Fitted preprocessor
└── checkpoint_epoch_*.pth  # Training checkpoints

mlruns/
└── [experiment_data]       # MLflow tracking

logs/
└── training.log            # Training logs
```

## 🔧 If You Still Get Errors

### Error: "FileNotFoundError: data/claims_train.parquet"
**Solution:**
```bash
python -c "import sys; sys.path.insert(0, '.'); from src.data_ingestion import load_sample_data; df = load_sample_data(10000); df.to_parquet('data/claims_train.parquet')"
```

### Error: "No module named 'src'"
**Solution:** Make sure you're in the project directory
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
python train.py --config config/example_config.yaml
```

### Error: CUDA/MPS issues
**Solution:** Force CPU usage in config
```yaml
training:
  accelerator: "cpu"
```

### Error: Out of memory
**Solution:** Reduce batch size
```yaml
training:
  batch_size: 128  # Or even 64
```

## 🎉 Quick Commands

```bash
# Full setup test
python test_setup.py

# Train with default config
python train.py --config config/example_config.yaml

# Train with starter config (simpler, 50 epochs)
python train.py --config config/starter_config.yaml

# View experiments
mlflow ui

# Run web app
streamlit run app.py
```

## 📈 Next Steps After Training

1. **Check Results:**
   ```bash
   mlflow ui
   # Open http://localhost:5000
   ```

2. **Score New Data:**
   ```bash
   python score.py --config config/example_config.yaml \
       --model-path models/best_model.pth \
       --preprocessor-path models/preprocessor.pkl \
       --input-path data/new_claims.parquet \
       --output-path results/scored.parquet
   ```

3. **Launch Dashboard:**
   ```bash
   streamlit run app.py
   ```

---

**Everything is now configured correctly!** 

Just run: `python train.py --config config/example_config.yaml`

🎉 **Happy Training!**
