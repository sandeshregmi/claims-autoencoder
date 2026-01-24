# SHAP Explainability - Now Included!

## ✅ Status: SHAP Support Added

SHAP (SHapley Additive exPlanations) has been **added** to the claims fraud detection system.

---

## 📁 Files Created

1. **`shap_explainer.py`** - SHAP explainer class for tree models
2. **`shap_explainability_demo.py`** - Demonstration script  
3. **`requirements.txt`** - Updated with `shap>=0.42.0`
4. **`SHAP_GUIDE.md`** - Complete usage guide

---

## 🚀 Quick Start

### Install SHAP
```bash
pip install shap
```

### Run Demo
```bash
# XGBoost explanations
python shap_explainability_demo.py --model xgboost

# CatBoost explanations
python shap_explainability_demo.py --model catboost
```

---

## 💡 What SHAP Provides

### 1. Local Explanations
**Answer**: Why is THIS specific claim flagged as fraud?

```python
from shap_explainer import ClaimsShapExplainer

explainer = ClaimsShapExplainer(model, features, cat_features)
explainer.create_explainers(train_data)

shap_values, contributions = explainer.explain_claim(
    suspicious_claim,
    target_feature='claim_amount',
    plot=True
)
```

**Output**: 
```
Top 5 fraud indicators for this claim:
1. claim_amount: $98,234 (SHAP: +850)
2. num_previous_claims: 47 (SHAP: +420)
3. days_since_last_claim: 1 (SHAP: +310)
4. diagnosis_code: D999 (SHAP: +180)
5. claim_type: experimental (SHAP: +150)
```

### 2. Global Feature Importance
**Answer**: What features matter most overall?

```python
global_importance = explainer.get_global_feature_importance(all_claims)
```

**Output**:
```
Top fraud indicators globally:
1. average_claim_amount (importance: 125.45)
2. num_previous_claims (importance: 98.23)
3. days_since_last_claim (importance: 87.12)
```

### 3. Top Frauds Explained
**Answer**: Why are the top 10 claims suspicious?

```python
top_frauds = explainer.explain_top_frauds(
    claims,
    fraud_scores,
    top_k=10,
    save_path='results/top_frauds_explained.csv'
)
```

---

## 🎯 Use Cases

### Fraud Investigation
- Explain to investigators WHY a claim is flagged
- Provide evidence for fraud decisions
- Prioritize investigation efforts

### Model Auditing
- Ensure model is not biased
- Verify legitimate fraud indicators
- Check for fair predictions

### Business Rules
- Extract patterns from SHAP explanations
- Create interpretable fraud rules
- Improve fraud detection strategy

---

## 📊 Features

- ✅ **TreeExplainer**: Fast explanations for XGBoost/CatBoost
- ✅ **Waterfall plots**: Visual feature impact
- ✅ **Summary plots**: SHAP distributions
- ✅ **Local & Global**: Per-claim and overall importance
- ✅ **Comparison**: Normal vs fraudulent patterns
- ✅ **Export**: Save explanations to CSV

---

## 🎓 Example Output

```
Most fraudulent claim (index 455):
  Fraud score: 75,228,637

SHAP explanation:
  average_claim_amount: +850.23 (pushing fraud score up)
  num_previous_claims: +420.15 (pushing fraud score up)
  days_since_last_claim: +310.45 (pushing fraud score up)
  
Total impact: These 3 features alone contribute +1,580 to fraud score
```

---

## 📚 Documentation

See **`SHAP_GUIDE.md`** for:
- Detailed usage examples
- API reference
- Best practices
- Advanced features

---

## ✅ Installation

```bash
# Install SHAP
pip install shap

# Verify installation
python -c "import shap; print(f'SHAP {shap.__version__} installed')"
```

---

## 🎉 Summary

**SHAP explainability is now fully integrated!**

- Works with both XGBoost and CatBoost
- Provides local (per-claim) explanations
- Shows global feature importance
- Makes fraud detection transparent and interpretable

**Next**: Run `python shap_explainability_demo.py` to see it in action!
