# FT-Transformer Not Available in Dashboard - Explanation

## Issue Found

The dashboard currently **only shows XGBoost and CatBoost** because it's using the `ClaimsTreeAutoencoder` class from `src/tree_models.py`, which only supports these two tree-based models.

## Why FT-Transformer is Missing

### Current Architecture:

```
Dashboard (webapp_enhanced.py)
    ↓
Uses: src/tree_models.py (ClaimsTreeAutoencoder)
    ↓
Supports ONLY: XGBoost, CatBoost
```

### FT-Transformer Location:

```
FT-Transformer exists in: /models/ft_transformer.py
But it's NOT integrated into the dashboard webapp!
```

## Available Models in Your Project

### ✅ Currently in Dashboard:
1. **XGBoost** - Fast tree-based model
2. **CatBoost** - Tree-based with categorical support

### ❌ NOT in Dashboard (but exist in code):
3. **FT-Transformer** - Deep learning transformer model
4. **TabNet** - Attention-based neural network
5. **ResNet** - Residual neural network
6. **MLP** - Multi-layer perceptron
7. **CLS Autoencoder** - Classification autoencoder

## Why Only Tree Models?

The dashboard was designed to use **tree-based models** for several reasons:

### Advantages of Tree Models (XGBoost/CatBoost):
- ✅ **Fast training** (seconds vs minutes)
- ✅ **No GPU required**
- ✅ **Works well on tabular data**
- ✅ **Native categorical support**
- ✅ **Easy to deploy**
- ✅ **Good interpretability**

### Disadvantages of Deep Learning Models (FT-Transformer):
- ❌ **Slower training** (requires many epochs)
- ❌ **Needs GPU for good performance**
- ❌ **More hyperparameters to tune**
- ❌ **Requires careful data preprocessing**
- ❌ **Harder to deploy**

## Solution Options

### Option 1: Use Tree Models (Recommended for Production)

**Current setup works great!** For claims fraud detection:
- XGBoost and CatBoost are industry standard
- Fast, reliable, and effective
- No need for deep learning complexity

**Action:** None needed - keep using the dashboard as-is

### Option 2: Add FT-Transformer to Dashboard (Advanced)

If you really need FT-Transformer, we would need to:

1. **Modify `webapp_enhanced.py`** to import FT-Transformer
2. **Add model selection** with 3 options: XGBoost, CatBoost, FT-Transformer
3. **Handle different training APIs** (tree vs neural network)
4. **Add GPU/CPU selection**
5. **Update training interface** for epochs, learning rate, etc.

This would be a **significant code change**.

### Option 3: Use Command-Line Training Script

For FT-Transformer training outside the dashboard:

```python
# train_ft_transformer.py
from models.ft_transformer import FTTransformer
from src.preprocessing import ClaimsPreprocessor
from src.data_ingestion import DataIngestion

# Load data
data_loader = DataIngestion(config)
train_df, val_df, test_df = data_loader.load_and_split()

# Preprocess
preprocessor = ClaimsPreprocessor()
X_train, y_train = preprocessor.fit_transform(train_df)

# Train FT-Transformer
model = FTTransformer(
    d_token=192,
    n_blocks=3,
    attention_dropout=0.2
)
model.fit(X_train, y_train, epochs=100)
```

You would run this separately from the dashboard.

## Recommendation

**For claims fraud detection, stick with XGBoost or CatBoost!**

Here's why:
- ✅ They're **proven effective** for tabular fraud detection
- ✅ **Much faster** to train and deploy
- ✅ **Easier to explain** to business stakeholders
- ✅ **Work well with small-medium datasets**
- ✅ **Don't require GPUs**

FT-Transformer is overkill unless you have:
- Very large datasets (millions of records)
- Complex feature interactions
- Access to GPUs for training

## Current Model Performance

Your current setup with tree models provides:
- **Feature importance** for interpretability
- **SHAP values** for explainability
- **Fast predictions** for real-time scoring
- **Good accuracy** on tabular data

## Summary

**Q: Why don't I see FT-Transformer?**
**A: The dashboard only uses tree-based models (by design)**

**Q: Where is FT-Transformer?**
**A: It exists in `/models/ft_transformer.py` but isn't integrated**

**Q: Should I add it?**
**A: For claims fraud, XGBoost/CatBoost are better choices**

**Q: Can I still use FT-Transformer?**
**A: Yes, but via Python scripts, not the dashboard**

---

**Bottom line: Your current setup with XGBoost and CatBoost is optimal for claims fraud detection!** 🎯
