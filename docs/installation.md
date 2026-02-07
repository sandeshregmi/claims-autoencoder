# Installation Guide

## Requirements

- Python 3.9+
- pip or conda

## Standard Installation

```bash
pip install claims-fraud
```

## Development Installation

```bash
git clone https://github.com/yourusername/claims-fraud.git
cd claims-fraud
pip install -e ".[dev]"
```

## Optional Dependencies

```bash
# For development
pip install claims-fraud[dev]

# For documentation
pip install claims-fraud[docs]
```

## Verify Installation

```bash
claims-fraud --version
python -c "import claims_fraud; print(claims_fraud.__version__)"
```
