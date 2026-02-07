# Claims Fraud Detection - Package Modularization Plan

## 🎯 Objective
Transform the current claims-autoencoder project into a portable, installable Python package with clean separation of concerns.

## 📋 Current State Analysis

### Current Structure Issues:
1. ❌ Monolithic webapp files (2000+ lines)
2. ❌ Mixed concerns (UI, business logic, data processing)
3. ❌ Scattered utility scripts and documentation
4. ❌ Hardcoded paths and configurations
5. ❌ No clear public API
6. ❌ Multiple duplicate/backup files

### Current Assets:
✅ Good src/ directory structure
✅ Existing setup.py foundation
✅ Separate config management
✅ Clear data pipeline components
✅ Well-defined models (tree_models, fairness, PSI)

---

## 🏗️ Proposed Package Structure

```
claims-fraud-detection/
│
├── pyproject.toml              # Modern Python packaging
├── setup.py                    # Backward compatibility
├── setup.cfg                   # Setup configuration
├── MANIFEST.in                 # Include non-Python files
├── README.md                   # Package documentation
├── LICENSE                     # License file
│
├── src/
│   └── claims_fraud/          # Main package (renamed for clarity)
│       │
│       ├── __init__.py        # Package initialization + version
│       ├── __version__.py     # Version info
│       │
│       ├── core/              # Core business logic
│       │   ├── __init__.py
│       │   ├── models.py      # Model definitions
│       │   ├── tree_models.py # Tree-based models
│       │   ├── scoring.py     # Fraud scoring logic
│       │   └── explainability.py # SHAP integration
│       │
│       ├── data/              # Data handling
│       │   ├── __init__.py
│       │   ├── ingestion.py   # Data loading
│       │   ├── preprocessing.py # Data preprocessing
│       │   └── validation.py  # Data validation
│       │
│       ├── analysis/          # Analysis modules
│       │   ├── __init__.py
│       │   ├── fairness.py    # Fairness analysis
│       │   ├── monitoring.py  # PSI monitoring
│       │   └── evaluation.py  # Model evaluation
│       │
│       ├── config/            # Configuration
│       │   ├── __init__.py
│       │   ├── manager.py     # Config management
│       │   ├── schemas.py     # Pydantic schemas
│       │   └── defaults.py    # Default configs
│       │
│       ├── ui/                # User Interface
│       │   ├── __init__.py
│       │   ├── app.py         # Main Streamlit app
│       │   ├── components/    # UI components
│       │   │   ├── __init__.py
│       │   │   ├── dashboard.py
│       │   │   ├── fraud_analysis.py
│       │   │   ├── fairness_tab.py
│       │   │   ├── monitoring_tab.py
│       │   │   └── shap_tab.py
│       │   └── utils/         # UI utilities
│       │       ├── __init__.py
│       │       ├── plots.py   # Plotting functions
│       │       └── formatters.py
│       │
│       ├── ml/                # Machine Learning
│       │   ├── __init__.py
│       │   ├── training.py    # Training pipeline
│       │   ├── tuning.py      # Hyperparameter tuning
│       │   └── registry.py    # Model registry
│       │
│       ├── utils/             # Utilities
│       │   ├── __init__.py
│       │   ├── logging.py     # Logging setup
│       │   ├── paths.py       # Path management
│       │   └── decorators.py  # Common decorators
│       │
│       └── cli/               # Command-line interface
│           ├── __init__.py
│           ├── train.py       # Training CLI
│           ├── score.py       # Scoring CLI
│           └── serve.py       # Web app CLI
│
├── configs/                   # Configuration files
│   ├── default.yaml
│   ├── development.yaml
│   └── production.yaml
│
├── data/                      # Data directory (gitignored)
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Pytest fixtures
│   ├── test_models/
│   ├── test_data/
│   ├── test_analysis/
│   └── test_ui/
│
├── examples/                  # Example scripts
│   ├── quickstart.py
│   ├── custom_model.py
│   └── batch_scoring.py
│
├── docs/                      # Documentation
│   ├── installation.md
│   ├── quickstart.md
│   ├── api/
│   └── tutorials/
│
└── scripts/                   # Development scripts
    ├── setup_env.sh
    ├── run_tests.sh
    └── build_package.sh
```

---

## 🔧 Implementation Steps

### Phase 1: Package Foundation (Priority: HIGH)
1. ✅ Create new package structure
2. ✅ Set up pyproject.toml with modern build system
3. ✅ Create proper __init__.py files
4. ✅ Define version management
5. ✅ Set up logging infrastructure

### Phase 2: Code Refactoring (Priority: HIGH)
6. ✅ Extract core business logic from webapp
7. ✅ Split webapp into components
8. ✅ Separate plotting/visualization functions
9. ✅ Create clean public API
10. ✅ Remove hardcoded paths

### Phase 3: Configuration (Priority: MEDIUM)
11. ✅ Centralize configuration management
12. ✅ Create configuration schemas with Pydantic
13. ✅ Support multiple environments
14. ✅ Add configuration validation

### Phase 4: CLI & Entry Points (Priority: MEDIUM)
15. ✅ Create CLI commands (train, score, serve)
16. ✅ Add proper argument parsing
17. ✅ Implement logging levels
18. ✅ Create user-friendly help messages

### Phase 5: Testing & Documentation (Priority: MEDIUM)
19. ✅ Set up pytest infrastructure
20. ✅ Write unit tests for core modules
21. ✅ Create integration tests
22. ✅ Write comprehensive README
23. ✅ Add API documentation

### Phase 6: Distribution (Priority: LOW)
24. ✅ Create wheel distribution
25. ✅ Test installation in clean environment
26. ✅ Create Docker image (optional)
27. ✅ Publish to PyPI (optional)

---

## 📦 Key Design Principles

### 1. Separation of Concerns
- **UI** should only handle presentation
- **Core** should contain business logic
- **Data** should handle I/O operations
- **ML** should manage model lifecycle

### 2. Dependency Injection
```python
# Bad: Hardcoded dependencies
class FraudDetector:
    def __init__(self):
        self.model = TreeModel()  # Hardcoded

# Good: Dependency injection
class FraudDetector:
    def __init__(self, model: BaseModel):
        self.model = model  # Injectable
```

### 3. Configuration as Code
```python
# Use Pydantic for validation
class ModelConfig(BaseModel):
    model_type: str = "catboost"
    n_estimators: int = 100
    learning_rate: float = 0.1
```

### 4. Clear Public API
```python
# Public API in __init__.py
from .core.scoring import FraudScorer
from .core.models import TreeModel
from .analysis.fairness import FairnessAnalyzer

__all__ = ['FraudScorer', 'TreeModel', 'FairnessAnalyzer']
```

### 5. Resource Management
```python
# Use context managers
with FraudScorer.from_config(config_path) as scorer:
    results = scorer.score(data)
```

---

## 🎨 Public API Design

### Core Classes
```python
# Main entry points
from claims_fraud import (
    FraudScorer,          # Scoring interface
    TreeModel,            # Model interface
    FairnessAnalyzer,     # Fairness analysis
    PSIMonitor,           # Drift monitoring
    DataPipeline,         # Data processing
)

# Quick start example
scorer = FraudScorer.from_pretrained("path/to/model")
scores = scorer.predict(data)
```

### CLI Commands
```bash
# Training
claims-fraud train --config configs/default.yaml --data data/train.parquet

# Scoring
claims-fraud score --model models/fraud_model.pkl --input data/test.parquet --output results.csv

# Web app
claims-fraud serve --port 8501 --config configs/production.yaml

# Evaluation
claims-fraud evaluate --model models/fraud_model.pkl --test-data data/test.parquet
```

---

## 🔄 Migration Strategy

### Step-by-Step Migration
1. **Create new structure** alongside existing code
2. **Gradually move** modules to new structure
3. **Update imports** progressively
4. **Run tests** after each module migration
5. **Keep old structure** until new one is validated
6. **Switch entry points** to new structure
7. **Archive old code** once migration complete

### Backward Compatibility
- Maintain old import paths temporarily
- Add deprecation warnings
- Provide migration guide
- Keep wrapper functions for 1-2 versions

---

## 📊 Benefits of Modularization

### For Development
✅ Easier to test individual components
✅ Faster development cycles
✅ Better code organization
✅ Easier onboarding for new developers
✅ Reduced merge conflicts

### For Deployment
✅ Installable via pip
✅ Version controlled
✅ Dependency management
✅ Environment reproducibility
✅ Easy CI/CD integration

### For Users
✅ Simple installation: `pip install claims-fraud`
✅ Clear API documentation
✅ CLI commands for common tasks
✅ Importable as library
✅ Customizable and extensible

---

## 🚀 Next Steps

### Immediate Actions:
1. Review and approve this plan
2. Create branch: `feature/modularization`
3. Set up new package structure
4. Begin Phase 1 implementation

### Success Criteria:
✅ Package installable via `pip install -e .`
✅ All tests passing
✅ CLI commands working
✅ Web app functional
✅ Documentation complete
✅ Clean public API

---

## 📝 Notes

### Dependencies to Keep
- Core: numpy, pandas, scikit-learn
- Models: xgboost, catboost, torch
- UI: streamlit, plotly
- Analysis: shap, scipy
- Config: pydantic, pyyaml

### Dependencies to Add
- click (for CLI)
- typer (alternative CLI framework)
- sphinx (for docs)
- pytest-cov (for coverage)
- black (for formatting)
- mypy (for type checking)

### Files to Archive/Remove
- Multiple webapp backups
- Temporary scripts (add_study_period*.py)
- Old documentation files
- Duplicate shell scripts

---

**Ready to proceed with implementation?**
