# Claims Fraud Detection - Clean Workflow Pipeline

A streamlined, production-ready end-to-end pipeline for claims fraud detection using tree-based models with explainability, monitoring, and fairness analysis.

## 🎯 Quick Start

```bash
# 1. Make scripts executable
chmod +x run_clean_workflow.sh clean_workflow.sh

# 2. (Optional) Clean up old files
./clean_workflow.sh

# 3. Run the application
./run_clean_workflow.sh
```

Access the dashboard at [http://localhost:8501](http://localhost:8501)

## 📁 Project Structure

```
claims-autoencoder/
├── src/
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
├── requirements_clean.txt        # Essential dependencies
├── run_clean_workflow.sh         # Main runner script
└── README_CLEAN.md               # This file
```

## 🔄 Workflow Pipeline

### 1. Data Preparation
Load and split the claims data into train/validation/test sets.

```python
from src.data_ingestion import DataIngestion
from src.config_manager import ConfigManager

config = ConfigManager("config/starter_config.yaml")
data_loader = DataIngestion(config)
train_df, val_df, test_df = data_loader.load_and_split()
```

### 2. Preprocessing
Clean, encode, and transform the data for model training.

```python
from src.preprocessing import ClaimsPreprocessor

preprocessor = ClaimsPreprocessor()
X_train, y_train = preprocessor.fit_transform(train_df)
X_val, y_val = preprocessor.transform(val_df)
```

### 3. Model Training
Train tree-based models (XGBoost or CatBoost) with automatic hyperparameter tuning.

```python
from src.tree_models import ClaimsTreeAutoencoder

model = ClaimsTreeAutoencoder(config)
model.train(X_train, y_train, X_val, y_val)
model.save("models/best_model.pkl")
```

### 4. Explainability
Generate SHAP values for model interpretability.

```python
from shap_explainer import ClaimsShapExplainer

explainer = ClaimsShapExplainer(model, X_train)
shap_values = explainer.compute_shap_values(X_test)
explainer.plot_summary()
```

### 5. Monitoring
Calculate PSI (Population Stability Index) to monitor data drift.

```python
from src.psi_monitoring import PSIMonitor

psi_monitor = PSIMonitor()
psi_scores = psi_monitor.calculate_psi(train_df, test_df)
```

### 6. Fairness Analysis
Evaluate model fairness across protected attributes.

```python
from src.fairness_analysis import FairnessAnalyzer

fairness = FairnessAnalyzer()
metrics = fairness.analyze(predictions, sensitive_attributes)
```

### 7. Web Dashboard
Interactive Streamlit dashboard with all features integrated.

```bash
streamlit run src/webapp_enhanced.py
```

## 🎨 Dashboard Features

### 📊 Overview Tab
- Key fraud statistics
- Model performance metrics
- Interactive visualizations

### 🔍 Predictions Tab
- Individual claim scoring
- Real-time fraud risk assessment
- Confidence scores

### 📈 Feature Importance
- Global feature importance
- SHAP waterfall plots
- Feature dependency plots

### 📉 PSI Monitoring
- Data drift detection
- Feature-level PSI scores
- Drift visualization

### ⚖️ Fairness Analysis
- Demographic parity
- Equal opportunity
- Disparate impact ratio

### 🔬 SHAP Analysis
- Force plots
- Summary plots
- Dependence plots
- Individual explanations

## ⚙️ Configuration

Edit `config/starter_config.yaml` to customize the pipeline:

```yaml
data:
  path: "data/claims_train.parquet"
  target_column: "fraud_flag"
  test_size: 0.2
  validation_size: 0.1

model:
  type: "xgboost"  # or "catboost"
  params:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
    scale_pos_weight: 10  # for imbalanced data

training:
  early_stopping_rounds: 10
  random_state: 42
  
monitoring:
  psi_threshold: 0.1
  fairness_threshold: 0.8
```

## 📦 Dependencies

Install with:
```bash
pip install -r requirements_clean.txt
```

Essential packages:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - ML utilities
- `xgboost` - Gradient boosting
- `catboost` - Gradient boosting (alternative)
- `shap` - Model explanations
- `streamlit` - Web application
- `plotly` - Interactive visualizations

## 🚀 Deployment Options

### Local Development
```bash
./run_clean_workflow.sh
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements_clean.txt
EXPOSE 8501
CMD ["streamlit", "run", "src/webapp_enhanced.py"]
```

### Cloud (AWS/GCP/Azure)
Deploy using container services or managed Streamlit hosting.

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

## 📝 Best Practices

1. **Data Quality**: Ensure data is properly validated before training
2. **Model Versioning**: Save models with timestamps
3. **Monitoring**: Regularly check PSI scores for data drift
4. **Fairness**: Monitor fairness metrics across different demographics
5. **Explainability**: Always provide SHAP explanations for predictions

## 🔧 Troubleshooting

### Common Issues

**Import Error for SHAP**
```bash
pip install --upgrade shap
```

**Streamlit Port Already in Use**
```bash
streamlit run src/webapp_enhanced.py --server.port 8502
```

**Data Loading Error**
- Verify file path in config
- Check parquet file is not corrupted
- Ensure proper file permissions

## 📚 Additional Resources

- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

This is a clean, production-ready pipeline. To contribute:
1. Keep the structure simple
2. Document all changes
3. Test thoroughly
4. Maintain backwards compatibility

## 📄 License

MIT License - See LICENSE file for details

## 🎓 Credits

Built with best practices in ML explainability, fairness, and monitoring.
