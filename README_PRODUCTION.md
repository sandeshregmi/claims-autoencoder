# Claims Fraud Detection - Production Ready

**AI-powered fraud detection system for insurance claims** using tree-based models (XGBoost/CatBoost), fairness analysis, and drift monitoring.

**Status:** ✅ Production Ready | **Version:** 0.1.0 | **License:** MIT

---

## 🚀 Quick Start

```bash
# 1. Install
make install

# 2. Start dashboard
make serve

# 3. Open browser
# → http://localhost:8501
```

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

### Core Capabilities
- **Tree-Based Models**: XGBoost and CatBoost for fast fraud detection
- **Per-Feature Autoencoders**: Reconstruction-based anomaly scoring
- **Feature Importance**: SHAP-based explanations for fraud decisions
- **Fairness Analysis**: Bias detection across demographic groups
- **PSI Monitoring**: Data drift detection and tracking
- **Interactive Dashboard**: Real-time fraud visualization and investigation

### Model Architecture
```
Input Claims Data
    ↓
[Preprocessing: Clean, Encode, Scale]
    ↓
[Per-Feature Autoencoders]
├─ Feature_1 Model → Error_1
├─ Feature_2 Model → Error_2
└─ Feature_N Model → Error_N
    ↓
[Fraud Score = Mean(Errors)]
    ↓
[Fairness Check & Bias Detection]
    ↓
[PSI Monitoring & Drift Alert]
```

---

## 📦 Installation

### Requirements
- Python 3.9+
- pip or conda

### Option 1: Package Installation

```bash
# Minimal (production)
make install

# Development (with testing tools)
make install-dev
```

### Option 2: From Source

```bash
# Clone repository
git clone https://github.com/your-org/claims-fraud.git
cd claims-fraud

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8 pylint
```

### Verify Installation

```bash
# Check package
python -c "import claims_fraud; print(claims_fraud.__version__)"

# Or run tests
make test
```

---

## 🎯 Usage

### 1. Interactive Dashboard

```bash
make serve
# Opens http://localhost:8501
```

**Dashboard Tabs:**
- **Overview**: Model performance, fraud statistics
- **Fraud Detection**: Real-time and batch scoring
- **Feature Importance**: SHAP explanations per claim
- **Fairness Analysis**: Bias detection across groups
- **Monitoring**: PSI drift tracking and alerts

### 2. Python API

```python
from claims_fraud import ClaimsTreeAutoencoder
from claims_fraud.data import DataIngestion

# Load data
ingestion = DataIngestion(config_path="config/config.yaml")
train_df, val_df, test_df = ingestion.load_and_split()

# Train model
model = ClaimsTreeAutoencoder(model_type="xgboost")
model.fit(train_df, categorical_features=["claim_type"], 
          numerical_features=["claim_amount"])

# Predict fraud scores
fraud_scores, errors = model.predict(test_df)
```

### 3. Command Line

```bash
# Start dashboard
claims-fraud serve

# Score batch of claims
claims-fraud score --input data/claims.parquet --output results/scored.parquet

# Train model
claims-fraud train --config config/config.yaml
```

### 4. Batch Scoring

```python
import pandas as pd
from claims_fraud import ClaimsTreeAutoencoder

# Load saved model
model = ClaimsTreeAutoencoder.load("models/production_model.pkl")

# Score new batch
df = pd.read_parquet("data/new_claims.parquet")
fraud_scores, errors = model.predict(df)

# Save results
df["fraud_score"] = fraud_scores
df.to_parquet("results/scored_claims.parquet")
```

---

## 🔧 Development

### Development Setup

```bash
# Install with dev dependencies
make install-dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Lint code
make lint
```

### Project Structure

```
claims-fraud/
├── src/claims_fraud/           # Main package
│   ├── core/                   # Model implementations
│   │   ├── tree_models.py      # XGBoost/CatBoost
│   │   ├── scoring.py          # Fraud scoring
│   │   └── explainability.py   # SHAP integration
│   ├── data/                   # Data handling
│   │   ├── ingestion.py        # Load & split
│   │   ├── preprocessing.py    # Feature engineering
│   │   └── validation.py       # Data quality
│   ├── analysis/               # Analysis modules
│   │   ├── evaluation.py       # Model metrics
│   │   ├── fairness.py         # Bias detection
│   │   └── monitoring.py       # PSI/drift
│   ├── config/                 # Configuration
│   │   ├── manager.py          # Config loading
│   │   └── schemas.py          # Validation
│   ├── cli/                    # Command line
│   │   ├── serve.py            # Dashboard
│   │   ├── train.py            # Training
│   │   ├── score.py            # Batch scoring
│   │   └── evaluate.py         # Evaluation
│   └── ui/                     # Streamlit app
│       └── app.py              # Dashboard
├── tests/                      # Test suite
│   ├── test_core/
│   ├── test_data/
│   └── test_analysis/
├── config/                     # Configuration files
│   └── config.yaml
├── data/                       # Data directory
│   └── claims_train.parquet    # Training data
├── models/                     # Saved models
├── setup.py                    # Package config
├── pyproject.toml              # Modern packaging
├── Makefile                    # Development tasks
└── README.md                   # This file
```

### Creating Tests

```python
# tests/test_core/test_models.py
import pytest
from claims_fraud.core.tree_models import ClaimsTreeAutoencoder

def test_model_training():
    model = ClaimsTreeAutoencoder(model_type="xgboost")
    # Your test here
    assert model is not None
```

---

## 🚢 Deployment

### Local Deployment

```bash
# Build package
make build

# Output: dist/claims-fraud-0.1.0-py3-none-any.whl
```

### Databricks Deployment

```bash
# 1. Configure Databricks CLI
make dbc-init

# 2. Validate bundle
make validate

# 3. Deploy to DEV
make deploy-dev

# 4. Deploy to PROD (requires confirmation)
make deploy-prod
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install package
COPY . .
RUN pip install .

# Expose dashboard port
EXPOSE 8501

# Run dashboard
CMD ["streamlit", "run", "src/claims_fraud/ui/app.py"]
```

```bash
# Build image
docker build -t claims-fraud:0.1.0 .

# Run container
docker run -p 8501:8501 claims-fraud:0.1.0
```

### Cloud Deployment (AWS)

```bash
# Deploy to AWS Lambda
aws lambda create-function \
  --function-name claims-fraud-scorer \
  --runtime python3.10 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-role \
  --handler claims_fraud.cli.score.lambda_handler \
  --zip-file fileb://dist/claims-fraud-0.1.0-py3-none-any.zip
```

---

## 📖 API Reference

### Main Classes

#### `ClaimsTreeAutoencoder`

```python
from claims_fraud.core.tree_models import ClaimsTreeAutoencoder

# Initialize
model = ClaimsTreeAutoencoder(
    model_type="xgboost",        # or "catboost"
    encoding_dim=64,
    model_params={...}
)

# Train
model.fit(
    data=train_df,
    categorical_features=["claim_type"],
    numerical_features=["claim_amount"]
)

# Predict
fraud_scores, per_feature_errors = model.predict(test_df)

# Save/Load
model.save("models/model.pkl")
model = ClaimsTreeAutoencoder.load("models/model.pkl")
```

#### `DataIngestion`

```python
from claims_fraud.data.ingestion import DataIngestion

ingestion = DataIngestion(config_path="config/config.yaml")

# Load and split
train_df, val_df, test_df = ingestion.load_and_split(
    train_size=0.7,
    val_size=0.15,
    test_size=0.15
)

# Load raw data
df = ingestion.load_raw_data("data/claims.parquet")
```

#### `FairnessAnalyzer`

```python
from claims_fraud.analysis.fairness import FairnessAnalyzer

analyzer = FairnessAnalyzer(
    fairness_groups=["age_group", "region"]
)

results = analyzer.analyze(
    data=test_df,
    predictions=fraud_scores
)

# Results include:
# - disparate_impact_ratio
# - flag_rate_parity
# - false_positive_rate_parity
```

#### `PSIMonitor`

```python
from claims_fraud.analysis.monitoring import PSIMonitor

monitor = PSIMonitor(reference_data=train_df)

psi_scores = monitor.calculate_psi(current_data=new_df)

# Alert if PSI > threshold
if psi_scores["claim_amount"] > 0.25:
    print("⚠️ Data drift detected!")
```

---

## ⚙️ Configuration

### Configuration File: `config/config.yaml`

```yaml
data:
  train_path: "data/claims_train.parquet"
  val_path: null  # Auto-split from train if null
  test_path: null

model:
  type: "xgboost"  # or "catboost"
  encoding_dim: 64
  hidden_layers: [128, 64]
  
features:
  categorical:
    - claim_type
    - provider_type
    - patient_gender
  numerical:
    - claim_amount
    - num_procedures
    - days_in_hospital

preprocessing:
  missing_value_strategy: "mean"  # or "median", "mode"
  outlier_method: "iqr"  # or "zscore"
  scaling: "standard"  # or "minmax"

fairness:
  protected_attributes:
    - age_group
    - region
  thresholds:
    disparate_impact: 0.80

monitoring:
  psi_threshold: 0.25
  check_interval_hours: 24
```

### Python Configuration

```python
from claims_fraud.config import load_config

config = load_config("config/config.yaml")

# Access values
model_type = config.model.type
train_path = config.data.train_path
```

---

## 🐛 Troubleshooting

### Issue: Dashboard won't start

```bash
# Clear cache
make cache-clear

# Reinstall
make install

# Start dashboard
make serve
```

### Issue: Model training fails

```bash
# Check data
python -c "import pandas as pd; df = pd.read_parquet('data/claims_train.parquet'); print(df.info())"

# Verify config
python -c "from claims_fraud.config import load_config; load_config('config/config.yaml')"

# Run tests
make test
```

### Issue: Deployment fails

```bash
# Validate Databricks config
make validate

# Check Databricks connection
make dbc-status

# View bundle details
databricks bundle show
```

### Issue: Memory errors on large datasets

```python
# Process in batches
BATCH_SIZE = 10000
for i in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[i:i+BATCH_SIZE]
    scores, errors = model.predict(batch)
    results.append((batch.index, scores))
```

### Issue: Low model performance

1. **Check data quality**: `make test-data`
2. **Review feature engineering**: Edit `config/config.yaml`
3. **Retrain model**: `make train`
4. **Analyze fairness**: Check dashboard Fairness tab

---

## 📊 Model Performance

### Metrics Tracked

- **Reconstruction Error**: Per-feature reconstruction loss
- **Fraud Detection Rate (TPR)**: % of actual fraud caught
- **False Positive Rate (FPR)**: % of legit claims flagged
- **AUC-ROC**: Area under ROC curve
- **Fairness Metrics**: Disparate impact ratio, parity gap
- **Data Drift (PSI)**: Population stability index

### Baseline Performance

On `claims_train.parquet` (5K claims):
- **Model**: XGBoost per-feature autoencoders
- **Train Time**: ~2 minutes
- **Inference Time**: ~50ms per 100 claims
- **AUC-ROC**: 0.87-0.92 (depending on features)
- **False Positive Rate**: 3-5%

---

## 🤝 Contributing

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes and test
make test
make lint
make format

# 3. Commit and push
git add .
git commit -m "feat: Add feature description"
git push origin feature/my-feature

# 4. Create pull request
# → https://github.com/your-org/claims-fraud/pull/new/feature/my-feature
```

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---

## 🆘 Support

### Quick Help

```bash
# Show all available commands
make

# Show environment info
make env-info

# Run tests
make test

# Clear cache
make cache-clear
```

### Resources

- **Documentation**: Check [src/claims_fraud/](src/claims_fraud/) module docstrings
- **Examples**: See [tests/](tests/) for usage examples
- **Issues**: Report via GitHub issues
- **Discussions**: Use GitHub discussions for questions

---

## 📈 Roadmap

### v0.2.0 (Q1 2026)
- [ ] Add LightGBM support
- [ ] Implement active learning for labeling
- [ ] Add REST API with FastAPI

### v0.3.0 (Q2 2026)
- [ ] Support for time-series features
- [ ] Ensemble meta-learner
- [ ] Advanced fairness constraints

### v1.0.0 (Q3 2026)
- [ ] Production hardening
- [ ] Multi-tenant support
- [ ] Real-time streaming ingestion

---

## 📜 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

**Last Updated:** February 15, 2026  
**Maintainer:** Claims Fraud Detection Team
