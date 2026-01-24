# Claims Autoencoder System 🏥

A production-ready anomaly detection system for insurance claims using deep learning autoencoders.

## 🎯 Features

- **Advanced Architecture**: Configurable autoencoder with multiple hidden layers and dropout
- **Robust Data Pipeline**: Handles missing values, outliers, and mixed data types
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Production Monitoring**: PSI (Population Stability Index) drift detection
- **Hyperparameter Tuning**: Optuna-based optimization
- **Interactive Dashboard**: Streamlit web interface for inference and monitoring
- **Batch Scoring**: Efficient processing of large datasets

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd claims-autoencoder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config/example_config.yaml` to customize:
- Data paths and formats
- Model architecture
- Training parameters
- Feature engineering rules

### Training

```bash
python src/training.py --config config/example_config.yaml
```

### Batch Scoring

```bash
python src/batch_scoring.py \
    --model-path models/best_model.pth \
    --input-path data/claims_to_score.parquet \
    --output-path results/scored_claims.parquet
```

### Web Interface

```bash
streamlit run src/webapp.py
```

Access at `http://localhost:8501`

## 📁 Project Structure

```
claims-autoencoder/
├── src/
│   ├── __init__.py
│   ├── config_manager.py       # Configuration handling
│   ├── data_ingestion.py       # Data loading utilities
│   ├── preprocessing.py        # Feature engineering
│   ├── model_architecture.py   # Autoencoder model
│   ├── training.py            # Training pipeline
│   ├── evaluation.py          # Model evaluation
│   ├── model_registry.py      # MLflow model management
│   ├── batch_scoring.py       # Batch inference
│   ├── psi_monitoring.py      # Drift detection
│   ├── webapp.py              # Streamlit dashboard
│   └── hyperparameter_tuning.py  # Optuna tuning
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_*.py
├── config/
│   └── example_config.yaml
├── docs/
├── requirements.txt
└── README.md
```

## 🔧 Configuration

The system is highly configurable via YAML. Key sections:

```yaml
data:
  train_path: "data/claims_train.parquet"
  numerical_features: [...]
  categorical_features: [...]

model:
  encoding_dim: 32
  hidden_layers: [128, 64]
  dropout_rate: 0.3

training:
  batch_size: 256
  learning_rate: 0.001
  max_epochs: 100
```

## 📊 Model Architecture

The autoencoder uses:
- **Encoder**: Compresses input features to low-dimensional representation
- **Decoder**: Reconstructs original features from encoded representation
- **Loss Function**: MSE for reconstruction error (anomaly score)

Anomalies are detected when reconstruction error exceeds a threshold (typically 95th percentile).

## 🔍 Monitoring

The system includes PSI monitoring to detect data drift:

```python
from src.psi_monitoring import PSIMonitor

monitor = PSIMonitor(reference_data, num_bins=10)
psi_scores = monitor.calculate_psi(new_data)
```

PSI thresholds:
- < 0.1: No significant change
- 0.1-0.2: Minor change
- \> 0.2: Major change (retrain recommended)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 🎛️ Hyperparameter Tuning

```python
from src.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(config, train_data, val_data)
best_params = tuner.optimize(n_trials=50)
```

## 📈 MLflow Tracking

View experiments:

```bash
mlflow ui
```

Access at `http://localhost:5000`

## 🔒 Security

- Input validation and sanitization
- Secure file upload handling
- Rate limiting (1000 requests/day per user)
- No sensitive data in logs

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- PyTorch Lightning for training framework
- MLflow for experiment tracking
- Streamlit for web interface
- Optuna for hyperparameter optimization
