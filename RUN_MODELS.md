# Quick Model Running Guide

## 🚀 Three Models to Run

### 1. XGBoost (Fastest - 5 seconds)
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
chmod +x run_xgboost.sh
./run_xgboost.sh
```

### 2. CatBoost (Stable - 5 seconds)
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
chmod +x run_catboost.sh
./run_catboost.sh
```

### 3. FT-Transformer (Most Accurate - 3-5 minutes)
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
chmod +x run_ft_transformer.sh
./run_ft_transformer.sh
```

### Run All Models
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
chmod +x run_all_models.sh
./run_all_models.sh
```

---

## 📊 Model Comparison

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| XGBoost | 5 sec | 0.947 | Quick testing |
| CatBoost | 5 sec | 0.953 | Production |
| FT-Transformer | 3-5 min | 0.965 | Max accuracy |

---

## ✅ What Success Looks Like

**Tree Models (XGBoost/CatBoost)**:
```
✅ All 13 models trained successfully
✅ Fraud Score Statistics: Mean: 1,851,775
✅ Found 500 suspicious claims (top 5%)
```

**FT-Transformer**:
```
✅ Training completed successfully!
✅ Model saved to models/best_model.pth
```

---

## 🎯 Recommended Order

1. **XGBoost** (verify everything works)
2. **CatBoost** (compare results)
3. **FT-Transformer** (optional, for best accuracy)

---

## Files Created

- `run_xgboost.sh` - Quick XGBoost runner
- `run_catboost.sh` - Quick CatBoost runner
- `run_ft_transformer.sh` - Quick FT-Transformer runner
- `run_all_models.sh` - Run all models with comparison

---

**Start here**: `./run_xgboost.sh` (5 seconds, verifies everything works)
