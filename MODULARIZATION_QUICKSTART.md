# 🚀 Quick Modularization Guide

## Overview
This guide will help you transform the `claims-autoencoder` project into a portable, professional Python package called `claims-fraud`.

---

## ⚡ Quick Start

### Option 1: Full Modularization (Recommended)
```bash
# Run complete modularization
python modularize_complete.py
```

### Option 2: Preview Changes First
```bash
# Dry run to see what will happen
python modularize_complete.py --dry-run
```

### Option 3: Phase-by-Phase
```bash
# Run specific phase
python modularize_complete.py --phase 1
python modularize_complete.py --phase 2
# ... etc
```

---

## 📋 What Will Happen

### 1. **Backup Created** (Phase 0)
- All existing code backed up to `_modularization_backup_TIMESTAMP/`
- Safe to experiment - can always restore

### 2. **New Structure Created** (Phase 1)
```
src/claims_fraud/
├── core/           # Core business logic
├── data/           # Data handling
├── analysis/       # Fairness, PSI, evaluation
├── ml/             # Training, tuning
├── config/         # Configuration
├── ui/             # Streamlit app (modularized)
├── utils/          # Utilities
└── cli/            # Command-line tools
```

### 3. **Files Created** (Phase 2)
- `pyproject.toml` - Modern packaging
- `__init__.py` - Public API
- `__version__.py` - Version management
- `MANIFEST.in` - Package data

### 4. **Code Migrated** (Phase 3)
- Existing modules moved to new structure
- Imports automatically updated
- Original files preserved in backup

### 5. **CLI Created** (Phase 4)
```bash
claims-fraud train --config config.yaml --data train.parquet
claims-fraud score --model model.pkl --input test.parquet
claims-fraud serve --port 8501
```

### 6. **Utilities Added** (Phase 5)
- Logging configuration
- Path management
- Common decorators

### 7. **Examples Created** (Phase 6)
- Quickstart guide
- Batch scoring example
- Custom model example

### 8. **Documentation** (Phase 7)
- README.md
- Installation guide
- API reference

---

## 🎯 After Modularization

### 1. Install the Package
```bash
# Development installation
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
pip install -e .
```

### 2. Verify Installation
```bash
# Check version
claims-fraud --version

# Python import
python -c "import claims_fraud; print(claims_fraud.__version__)"
```

### 3. Use the Package

#### As a Library
```python
from claims_fraud import FraudDetector, TreeModel

# Train
model = TreeModel(model_type="catboost")
model.fit(data, categorical_features=['type'], 
          numerical_features=['amount'])

# Detect
detector = FraudDetector(model)
scores = detector.predict(new_claims)
```

#### As CLI
```bash
# Train
claims-fraud train \
    --config configs/default.yaml \
    --data data/train.parquet \
    --output models/fraud_model.pkl

# Score
claims-fraud score \
    --model models/fraud_model.pkl \
    --input data/test.parquet \
    --output results/scores.csv

# Dashboard
claims-fraud serve --port 8501
```

---

## 🔍 Key Changes

### Package Name
- **Old**: `claims-autoencoder`
- **New**: `claims-fraud`
- **Import**: `import claims_fraud`

### Module Organization
- **Before**: Flat structure in `src/`
- **After**: Organized by functionality

### Imports
- **Before**: `from src.tree_models import ...`
- **After**: `from claims_fraud.core.tree_models import ...`

### Entry Points
- **Before**: Direct script execution
- **After**: Professional CLI commands

---

## 📁 File Locations

### Key Files
| Purpose | Location |
|---------|----------|
| Main package | `src/claims_fraud/__init__.py` |
| Version | `src/claims_fraud/__version__.py` |
| Packaging | `pyproject.toml` |
| CLI | `src/claims_fraud/cli/` |
| Models | `src/claims_fraud/core/` |
| Web UI | `src/claims_fraud/ui/` |
| Tests | `tests/` |
| Examples | `examples/` |
| Docs | `docs/` |

### Old vs New
| Old Location | New Location |
|--------------|--------------|
| `src/tree_models.py` | `src/claims_fraud/core/tree_models.py` |
| `src/fairness_analysis.py` | `src/claims_fraud/analysis/fairness.py` |
| `src/webapp_enhanced.py` | `src/claims_fraud/ui/app.py` |
| `src/config_manager.py` | `src/claims_fraud/config/manager.py` |

---

## 🛠️ Development Workflow

### Setup Development Environment
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Run Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=claims_fraud --cov-report=html

# Specific test
pytest tests/test_core/test_models.py
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/
```

### Build Package
```bash
# Install build tools
pip install build

# Build distribution
python -m build

# Result: dist/claims_fraud-0.1.0-py3-none-any.whl
```

---

## 🚨 Troubleshooting

### "Module not found: claims_fraud"
```bash
# Ensure installation
pip install -e .

# Check PYTHONPATH
python -c "import sys; print('\n'.join(sys.path))"
```

### "Command not found: claims-fraud"
```bash
# Reinstall with entry points
pip uninstall claims-fraud
pip install -e .

# Check installation
pip show claims-fraud
```

### Import errors after migration
```bash
# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Reinstall
pip install -e . --force-reinstall
```

---

## 🎨 Customization

### Change Package Name
Edit `pyproject.toml`:
```toml
[project]
name = "your-package-name"
```

### Add Dependencies
Edit `pyproject.toml`:
```toml
dependencies = [
    "existing-deps>=1.0.0",
    "your-new-dep>=2.0.0",
]
```

### Add CLI Commands
Create new file in `src/claims_fraud/cli/`:
```python
import click

@click.command()
def your_command():
    """Your command description"""
    pass
```

Register in `cli/__init__.py`:
```python
main.add_command(your_command)
```

---

## 📊 Checklist

### Before Running
- [ ] Code committed to git
- [ ] Virtual environment activated
- [ ] Requirements installed

### After Running
- [ ] Backup created successfully
- [ ] New structure created
- [ ] Package installs: `pip install -e .`
- [ ] CLI works: `claims-fraud --help`
- [ ] Imports work: `import claims_fraud`
- [ ] Web app runs: `claims-fraud serve`
- [ ] Tests pass: `pytest`

---

## 🆘 Support

### Rollback
If something goes wrong:
```bash
# Restore from backup
cp -r _modularization_backup_TIMESTAMP/src .
```

### Check Logs
```bash
# See what was created
find src/claims_fraud -type f -name "*.py" | head -20
```

### Get Help
1. Check `MODULARIZATION_EXECUTION_PLAN.md`
2. Review phase-specific logs
3. Check backup directory
4. Restore and try again

---

## ✅ Success Indicators

After successful modularization:
```bash
# ✅ Package installed
pip list | grep claims-fraud

# ✅ CLI available
claims-fraud --version

# ✅ Python import works
python -c "import claims_fraud; print('Success!')"

# ✅ All commands available
claims-fraud --help

# ✅ Web app works
claims-fraud serve  # Opens on http://localhost:8501
```

---

**Ready to modularize?**

```bash
python modularize_complete.py
```

**Questions?** See `MODULARIZATION_EXECUTION_PLAN.md` for detailed information.
