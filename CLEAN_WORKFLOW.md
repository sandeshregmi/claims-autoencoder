# Clean End-to-End Workflow Pipeline

## Core Files Structure

```
claims-autoencoder/
├── src/
│   ├── __init__.py
│   ├── webapp_enhanced.py          # Main web application
│   ├── config_manager.py           # Configuration management
│   ├── data_ingestion.py          # Data loading utilities
│   ├── preprocessing.py           # Data preprocessing
│   ├── tree_models.py             # Tree-based models (XGBoost, CatBoost)
│   ├── psi_monitoring.py          # PSI drift monitoring
│   └── fairness_analysis.py       # Fairness metrics
├── config/
│   └── starter_config.yaml        # Base configuration
├── data/
│   └── claims_train.parquet       # Training data
├── models/                        # Saved trained models
├── shap_explainer.py             # SHAP explanations
├── requirements.txt              # Python dependencies
├── README.md                     # Main documentation
└── QUICKSTART.md                 # Quick start guide
```

## Workflow Pipeline

### 1. Data Preparation
```python
from src.data_ingestion import DataIngestion
from src.config_manager import ConfigManager

config = ConfigManager("config/starter_config.yaml")
data_loader = DataIngestion(config)
train_df, val_df, test_df = data_loader.load_and_split()
```

### 2. Preprocessing
```python
from src.preprocessing import ClaimsPreprocessor

preprocessor = ClaimsPreprocessor()
X_train, y_train = preprocessor.fit_transform(train_df)
X_val, y_val = preprocessor.transform(val_df)
```

### 3. Model Training
```python
from src.tree_models import ClaimsTreeAutoencoder

model = ClaimsTreeAutoencoder(config)
model.train(X_train, y_train, X_val, y_val)
model.save("models/best_model.pkl")
```

### 4. Explainability
```python
from shap_explainer import ClaimsShapExplainer

explainer = ClaimsShapExplainer(model, X_train)
shap_values = explainer.compute_shap_values(X_test)
explainer.plot_summary()
```

### 5. Monitoring
```python
from src.psi_monitoring import PSIMonitor

psi_monitor = PSIMonitor()
psi_scores = psi_monitor.calculate_psi(train_df, test_df)
```

### 6. Fairness Analysis
```python
from src.fairness_analysis import FairnessAnalyzer

fairness = FairnessAnalyzer()
metrics = fairness.analyze(predictions, sensitive_attributes)
```

### 7. Web Dashboard
```bash
streamlit run src/webapp_enhanced.py
```

## Files to Remove

### Documentation Files (Outdated)
- All `*_COMPLETE.md` files
- All `*_FIX.md` files  
- All `*_GUIDE.md` files (except QUICKSTART.md)
- `*_IMPLEMENTATION.md` files
- `*_ARCHITECTURE.md` files
- `*_CHANGELOG.md` files

### Duplicate Application Files
- `app.py` (if webapp_enhanced.py is main)
- `app_complete.py`
- `app_enhanced.py`
- `src/webapp.py` (old version)
- `src/webapp_enhanced_COMPLETE.py`
- `src/webapp_enhanced_backup_*.py`

### Utility Scripts (One-time use)
- `add_*.py`
- `apply_*.py`
- `create_*.py`
- `upgrade_*.py`
- `fix_*.sh`
- `cleanup_*.sh`

### Test Files at Root
- `test.py`
- `test_*.py` (move to tests/ directory)

### Build/Cache Directories
- `__pycache__/` (all)
- `.pytest_cache/`
- `catboost_info/`
- `.DS_Store` files

### Checkpoints (Keep only best)
- `checkpoints/checkpoint_epoch_*.pth` (remove all except best)

### Databricks (if not used)
- `.databricks/` directory
- `databricks.yml`
- `databricks_*.py` files
- `deploy_databricks.sh`

## Quick Start After Cleanup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web app
streamlit run src/webapp_enhanced.py

# 3. Access at http://localhost:8501
```

## Minimal Configuration

```yaml
# config/starter_config.yaml
data:
  path: "data/claims_train.parquet"
  target_column: "fraud_flag"

model:
  type: "xgboost"
  params:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100

training:
  validation_split: 0.2
  random_state: 42
```
