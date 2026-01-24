.PHONY: help install install-dev test train score webapp clean

help:
	@echo "Claims Autoencoder - Available Commands"
	@echo "========================================"
	@echo "make install       - Install package in production mode"
	@echo "make install-dev   - Install package in development mode"
	@echo "make test          - Run test suite"
	@echo "make test-cov      - Run tests with coverage"
	@echo "make sample-data   - Generate sample data"
	@echo "make train         - Train model with default config"
	@echo "make score         - Score claims (requires MODEL_PATH, INPUT_PATH)"
	@echo "make webapp        - Launch Streamlit dashboard"
	@echo "make mlflow        - Start MLflow UI"
	@echo "make clean         - Clean generated files"
	@echo "make lint          - Run code linting"

install:
	pip install .

install-dev:
	pip install -e .
	pip install pytest pytest-cov black flake8 mypy

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

sample-data:
	python -c "import sys; sys.path.insert(0, '.'); from src.data_ingestion import load_sample_data; df = load_sample_data(10000); df.to_parquet('data/claims_train.parquet'); print('Sample data created in data/claims_train.parquet')"

train:
	python train.py --config config/example_config.yaml

score:
	@if [ -z "$(MODEL_PATH)" ] || [ -z "$(INPUT_PATH)" ]; then \
		echo "Error: Please set MODEL_PATH and INPUT_PATH"; \
		echo "Usage: make score MODEL_PATH=models/best_model.pth INPUT_PATH=data/claims.parquet OUTPUT_PATH=results/scored.parquet"; \
		exit 1; \
	fi
	python score.py --config config/example_config.yaml \
		--model-path $(MODEL_PATH) \
		--preprocessor-path models/preprocessor.pkl \
		--input-path $(INPUT_PATH) \
		--output-path $(or $(OUTPUT_PATH),results/scored_claims.parquet)

webapp:
	streamlit run app.py

mlflow:
	mlflow ui

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	flake8 src/ --max-line-length=100 --ignore=E203,W503
	black src/ tests/ --check
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

setup-dirs:
	mkdir -p data models outputs logs checkpoints results

full-setup: install-dev setup-dirs sample-data
	@echo "✅ Setup complete!"
	@echo "Run 'make train' to train your first model"
