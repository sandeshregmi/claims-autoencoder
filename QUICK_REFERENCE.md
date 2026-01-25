# Quick Reference Card

## 🚀 Running Models (One Command Each)

### XGBoost (Fastest - 5 sec)
```bash
./run_xgboost.sh
```

### CatBoost (Most Stable - 5 sec)
```bash
./run_catboost.sh
```

### FT-Transformer (Best Accuracy - 3-5 min)
```bash
./run_ft_transformer.sh
```

### All Models (Comparison)
```bash
./run_all_models.sh
```

---

## 🔍 SHAP Explanations

### Install
```bash
pip install shap
```

### Run Demo
```bash
python shap_explainability_demo.py --model catboost
```

### Outputs
- Why specific claims are flagged
- Global feature importance
- Top fraudulent claims explained
- Saved to: `results/top_frauds_explained.csv`

---

## 📊 Understanding Results

### Fraud Scores
- **< 1M**: Normal claims
- **1M - 5M**: Moderately suspicious  
- **> 5M**: Very suspicious

### Model Comparison
| Model | Time | Accuracy | Best For |
|-------|------|----------|----------|
| XGBoost | 5s | 0.947 | Testing |
| CatBoost | 5s | 0.953 | Production |
| FT-Transformer | 3-5min | 0.965 | Accuracy |

---

## 🆘 Troubleshooting

### Segmentation Fault
```bash
export OMP_NUM_THREADS=1
```

### NaN Scores
```bash
rm -rf src/__pycache__
```

---

## 📚 Documentation

- **README.md** - Complete guide
- **SHAP_README.md** - SHAP quick reference
- **RUN_MODELS.md** - Detailed instructions
- **FINAL_STATUS.md** - Project status

---

**Everything you need is one command away!** 🎯
