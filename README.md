# Claims Fraud Detection 🕵️

AI-powered fraud detection system for insurance claims using tree-based models, fairness analysis, and drift monitoring.

## 🚀 Quick Start

### Installation

```bash
pip install claims-fraud
```

### Basic Usage

```python
from claims_fraud import FraudDetector, TreeModel

# Create and train model
model = TreeModel(model_type="catboost")
model.fit(data, categorical_features=['claim_type'], 
          numerical_features=['claim_amount'])

# Detect fraud
detector = FraudDetector(model)
fraud_scores = detector.predict(new_claims)
```

### CLI Usage

```bash
# Train model
claims-fraud train --config config.yaml --data train.parquet

# Score claims
claims-fraud score --model model.pkl --input test.parquet --output scores.csv

# Launch dashboard
claims-fraud serve --port 8501
```

## 📦 Features

- **Fraud Detection**: XGBoost & CatBoost models
- **Fairness Analysis**: Bias detection across demographics
- **Drift Monitoring**: PSI-based model degradation detection
- **Explainability**: SHAP values for interpretability
- **Web Dashboard**: Interactive Streamlit interface
- **CLI Tools**: Command-line tools for all operations

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start](docs/quickstart.md)
- [API Reference](docs/api_reference.md)

## 🛠️ Development

```bash
# Clone & install
git clone https://github.com/yourusername/claims-fraud.git
cd claims-fraud
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.
