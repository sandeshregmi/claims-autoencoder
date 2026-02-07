# 🎯 Claims Fraud Detection - Complete Modularization Plan

## Current State Analysis

### ✅ Strengths
- Well-organized src/ directory
- Clear separation of concerns (data, models, analysis)
- Existing setup.py foundation
- Good module naming conventions

### ❌ Issues to Fix
1. **Monolithic webapp** (2000+ lines, needs splitting)
2. **Multiple backup files** (cleanup needed)
3. **Old package name** ("claims-autoencoder" → "claims-fraud")
4. **Missing modern packaging** (no pyproject.toml)
5. **No CLI framework** (hardcoded entry points)
6. **Hardcoded paths** throughout codebase
7. **No public API** (unclear what to import)
8. **Missing tests** for most modules

---

## 🏗️ Target Structure

```
claims-fraud-detection/
├── pyproject.toml          # Modern Python packaging (PEP 621)
├── setup.py                # Backward compatibility
├── setup.cfg               # Additional config
├── MANIFEST.in             # Package data
├── README.md               # Main documentation
├── LICENSE                 # MIT License
│
├── src/
│   └── claims_fraud/       # Renamed from claims-autoencoder
│       ├── __init__.py     # Public API
│       ├── __version__.py  # Version info
│       │
│       ├── core/           # Core business logic
│       │   ├── __init__.py
│       │   ├── base.py              # Base classes/interfaces
│       │   ├── tree_models.py       # Tree-based models
│       │   ├── scoring.py           # Fraud scoring engine
│       │   └── explainability.py    # SHAP wrapper
│       │
│       ├── data/           # Data management
│       │   ├── __init__.py
│       │   ├── ingestion.py         # Data loading
│       │   ├── preprocessing.py     # Data preprocessing
│       │   ├── validation.py        # Data validation
│       │   └── schemas.py           # Data schemas
│       │
│       ├── analysis/       # Analysis modules
│       │   ├── __init__.py
│       │   ├── fairness.py          # Fairness analysis
│       │   ├── monitoring.py        # PSI monitoring
│       │   └── evaluation.py        # Model evaluation
│       │
│       ├── ml/             # ML operations
│       │   ├── __init__.py
│       │   ├── training.py          # Training pipeline
│       │   ├── tuning.py            # Hyperparameter tuning
│       │   └── registry.py          # Model registry
│       │
│       ├── config/         # Configuration
│       │   ├── __init__.py
│       │   ├── manager.py           # Config manager
│       │   ├── schemas.py           # Pydantic schemas
│       │   └── defaults.yaml        # Default config
│       │
│       ├── ui/             # User interface
│       │   ├── __init__.py
│       │   ├── app.py               # Main Streamlit app
│       │   ├── components/          # UI components
│       │   │   ├── __init__.py
│       │   │   ├── header.py        # Header component
│       │   │   ├── sidebar.py       # Sidebar component
│       │   │   ├── dashboard_tab.py
│       │   │   ├── fraud_tab.py
│       │   │   ├── importance_tab.py
│       │   │   ├── analysis_tab.py
│       │   │   ├── shap_tab.py
│       │   │   ├── monitoring_tab.py
│       │   │   ├── fairness_tab.py
│       │   │   └── export_tab.py
│       │   └── utils/               # UI utilities
│       │       ├── __init__.py
│       │       ├── plots.py         # Plotting functions
│       │       ├── formatters.py    # Data formatters
│       │       └── state.py         # Session state
│       │
│       ├── utils/          # General utilities
│       │   ├── __init__.py
│       │   ├── logging.py           # Logging setup
│       │   ├── paths.py             # Path management
│       │   ├── decorators.py        # Common decorators
│       │   └── io.py                # I/O utilities
│       │
│       └── cli/            # Command-line interface
│           ├── __init__.py          # Main CLI
│           ├── train.py             # Training commands
│           ├── score.py             # Scoring commands
│           ├── evaluate.py          # Evaluation commands
│           └── serve.py             # Web app launcher
│
├── configs/                # Configuration files
│   ├── default.yaml
│   ├── development.yaml
│   └── production.yaml
│
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core/
│   ├── test_data/
│   ├── test_analysis/
│   ├── test_ml/
│   └── test_cli/
│
├── examples/               # Example scripts
│   ├── quickstart.py
│   ├── batch_scoring.py
│   ├── custom_model.py
│   └── fairness_analysis.py
│
├── docs/                   # Documentation
│   ├── installation.md
│   ├── quickstart.md
│   ├── user_guide.md
│   ├── api_reference.md
│   └── tutorials/
│
└── scripts/                # Development scripts
    ├── setup_dev.sh
    ├── run_tests.sh
    ├── build_package.sh
    └── clean_cache.sh
```

---

## 📋 Execution Steps

### Phase 1: Preparation ✅
- [x] Analyze current structure
- [x] Create modularization plan
- [x] Create backup strategy
- [ ] Execute: Run step 1

### Phase 2: Structure Setup
- [ ] Create new directory structure
- [ ] Set up pyproject.toml
- [ ] Create __init__.py files
- [ ] Set up version management
- [ ] Execute: Run step 2

### Phase 3: Code Migration
- [ ] Migrate core models
- [ ] Migrate data modules
- [ ] Migrate analysis modules
- [ ] Migrate ML operations
- [ ] Update all imports
- [ ] Execute: Run step 3

### Phase 4: Webapp Modularization
- [ ] Extract header component
- [ ] Extract sidebar logic
- [ ] Split into tab components
- [ ] Extract plotting functions
- [ ] Create state management
- [ ] Execute: Run step 4

### Phase 5: CLI & Entry Points
- [ ] Create CLI framework
- [ ] Implement train command
- [ ] Implement score command
- [ ] Implement serve command
- [ ] Update entry points
- [ ] Execute: Run step 5

### Phase 6: Configuration
- [ ] Centralize config management
- [ ] Create Pydantic schemas
- [ ] Set up environment configs
- [ ] Add config validation
- [ ] Execute: Run step 6

### Phase 7: Testing
- [ ] Set up pytest
- [ ] Write core tests
- [ ] Write integration tests
- [ ] Set up coverage
- [ ] Execute: Run step 7

### Phase 8: Documentation
- [ ] Update README
- [ ] Write installation guide
- [ ] Create API docs
- [ ] Add examples
- [ ] Execute: Run step 8

### Phase 9: Cleanup
- [ ] Remove backup files
- [ ] Clean old scripts
- [ ] Archive deprecated code
- [ ] Update .gitignore
- [ ] Execute: Run step 9

### Phase 10: Validation
- [ ] Test installation
- [ ] Run all tests
- [ ] Test CLI commands
- [ ] Test web app
- [ ] Validate distribution
- [ ] Execute: Run step 10

---

## 🎯 Key Design Decisions

### 1. Package Naming
- **Old**: `claims-autoencoder`
- **New**: `claims-fraud`
- **Import**: `import claims_fraud`

### 2. Public API
```python
# Top-level imports
from claims_fraud import (
    FraudDetector,      # Main detector class
    TreeModel,          # Model wrapper
    FairnessAnalyzer,   # Fairness analysis
    PSIMonitor,         # Drift monitoring
    DataPipeline,       # Data processing
)
```

### 3. CLI Design
```bash
claims-fraud train --config config.yaml --data train.parquet
claims-fraud score --model model.pkl --input test.parquet --output scores.csv
claims-fraud evaluate --model model.pkl --test test.parquet
claims-fraud serve --port 8501 --config config.yaml
```

### 4. Configuration Strategy
- YAML-based configuration
- Pydantic validation
- Environment-specific configs
- Override via CLI args

### 5. Import Strategy
```python
# Absolute imports only
from claims_fraud.core.tree_models import ClaimsTreeAutoencoder
from claims_fraud.data.ingestion import DataIngestion
from claims_fraud.analysis.fairness import FairnessAnalyzer

# No relative imports in public API
```

---

## 🚀 Quick Start Commands

```bash
# Step 1: Create structure
python modularize_step1_structure.py

# Step 2: Migrate code
python modularize_step2_migrate.py

# Step 3: Split webapp
python modularize_step3_webapp.py

# Step 4: Test installation
pip install -e .

# Step 5: Run tests
pytest

# Step 6: Build package
python -m build

# Step 7: Test distribution
pip install dist/claims_fraud-0.1.0-py3-none-any.whl
```

---

## ✅ Success Criteria

- [x] Package structure created
- [ ] All modules migrated
- [ ] Imports updated
- [ ] CLI working
- [ ] Tests passing
- [ ] Documentation complete
- [ ] Installation successful
- [ ] Web app functional
- [ ] Distribution builds

---

## 📊 Progress Tracking

| Phase | Status | Progress |
|-------|--------|----------|
| 1. Preparation | ✅ Complete | 100% |
| 2. Structure | 🔄 In Progress | 0% |
| 3. Migration | ⏳ Pending | 0% |
| 4. Webapp | ⏳ Pending | 0% |
| 5. CLI | ⏳ Pending | 0% |
| 6. Config | ⏳ Pending | 0% |
| 7. Testing | ⏳ Pending | 0% |
| 8. Docs | ⏳ Pending | 0% |
| 9. Cleanup | ⏳ Pending | 0% |
| 10. Validation | ⏳ Pending | 0% |

**Overall Progress: 10%**

---

**Ready to execute? Run:**
```bash
python modularize_complete.py
```
