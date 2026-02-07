#!/usr/bin/env python3
"""
Automated Package Modularization Script
Step 1: Create the new package structure
"""

import os
from pathlib import Path
import shutil
from datetime import datetime

class PackageModularizer:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.new_package_name = "claims_fraud"
        self.backup_dir = self.base_path / f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def create_backup(self):
        """Create backup of critical files"""
        print("📦 Creating backup...")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup src directory
        if (self.base_path / "src").exists():
            shutil.copytree(
                self.base_path / "src",
                self.backup_dir / "src",
                dirs_exist_ok=True
            )
        print(f"✅ Backup created at: {self.backup_dir}")
    
    def create_new_structure(self):
        """Create the new modular package structure"""
        print("\n🏗️  Creating new package structure...")
        
        # Define the new structure
        structure = {
            f"src/{self.new_package_name}": [
                "__init__.py",
                "__version__.py",
            ],
            f"src/{self.new_package_name}/core": [
                "__init__.py",
                "models.py",
                "tree_models.py",
                "scoring.py",
                "explainability.py",
            ],
            f"src/{self.new_package_name}/data": [
                "__init__.py",
                "ingestion.py",
                "preprocessing.py",
                "validation.py",
            ],
            f"src/{self.new_package_name}/analysis": [
                "__init__.py",
                "fairness.py",
                "monitoring.py",
                "evaluation.py",
            ],
            f"src/{self.new_package_name}/config": [
                "__init__.py",
                "manager.py",
                "schemas.py",
                "defaults.py",
            ],
            f"src/{self.new_package_name}/ui": [
                "__init__.py",
                "app.py",
            ],
            f"src/{self.new_package_name}/ui/components": [
                "__init__.py",
                "dashboard.py",
                "fraud_analysis.py",
                "fairness_tab.py",
                "monitoring_tab.py",
                "shap_tab.py",
            ],
            f"src/{self.new_package_name}/ui/utils": [
                "__init__.py",
                "plots.py",
                "formatters.py",
            ],
            f"src/{self.new_package_name}/ml": [
                "__init__.py",
                "training.py",
                "tuning.py",
                "registry.py",
            ],
            f"src/{self.new_package_name}/utils": [
                "__init__.py",
                "logging.py",
                "paths.py",
                "decorators.py",
            ],
            f"src/{self.new_package_name}/cli": [
                "__init__.py",
                "train.py",
                "score.py",
                "serve.py",
            ],
            "configs": [
                "default.yaml",
                "development.yaml",
                "production.yaml",
            ],
            "tests": [
                "__init__.py",
                "conftest.py",
            ],
            "tests/test_core": [
                "__init__.py",
                "test_models.py",
                "test_scoring.py",
            ],
            "tests/test_data": [
                "__init__.py",
                "test_ingestion.py",
            ],
            "tests/test_analysis": [
                "__init__.py",
                "test_fairness.py",
            ],
            "examples": [
                "quickstart.py",
                "batch_scoring.py",
            ],
            "docs": [
                "installation.md",
                "quickstart.md",
            ],
            "scripts": [
                "setup_env.sh",
                "run_tests.sh",
            ],
        }
        
        # Create directories and files
        for directory, files in structure.items():
            dir_path = self.base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  📁 Created: {directory}/")
            
            for file in files:
                file_path = dir_path / file
                if not file_path.exists():
                    file_path.touch()
                    print(f"    📄 Created: {directory}/{file}")
        
        print("✅ New structure created!")
    
    def create_version_file(self):
        """Create __version__.py"""
        version_content = '''"""Version information for claims_fraud package"""

__version__ = "0.1.0"
__author__ = "Claims Fraud Detection Team"
__license__ = "MIT"
'''
        version_path = self.base_path / f"src/{self.new_package_name}/__version__.py"
        version_path.write_text(version_content)
        print(f"✅ Created version file: {version_path}")
    
    def create_main_init(self):
        """Create main package __init__.py"""
        init_content = '''"""
Claims Fraud Detection Package

A comprehensive fraud detection system for insurance claims using
tree-based models, fairness analysis, and drift monitoring.
"""

from .__ version__ import __version__, __author__, __license__

# Core components
from .core.scoring import FraudScorer
from .core.tree_models import ClaimsTreeAutoencoder
from .analysis.fairness import FairnessAnalyzer
from .analysis.monitoring import PSIMonitor
from .data.ingestion import DataPipeline

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Core classes
    'FraudScorer',
    'ClaimsTreeAutoencoder',
    'FairnessAnalyzer',
    'PSIMonitor',
    'DataPipeline',
]
'''
        init_path = self.base_path / f"src/{self.new_package_name}/__init__.py"
        init_path.write_text(init_content)
        print(f"✅ Created main __init__.py: {init_path}")
    
    def create_pyproject_toml(self):
        """Create modern pyproject.toml"""
        pyproject_content = '''[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "claims-fraud"
version = "0.1.0"
description = "AI-powered fraud detection system for insurance claims"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Claims Fraud Detection Team"}
]
keywords = ["fraud-detection", "insurance", "machine-learning", "anomaly-detection"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "numpy>=1.24.0,<2.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "catboost>=1.2.0",
    "torch>=2.0.0",
    "pytorch-tabnet>=4.0",
    "shap>=0.44.0",
    "scipy>=1.11.0",
    "streamlit>=1.28.0",
    "plotly>=5.17.0",
    "seaborn>=0.12.0",
    "matplotlib>=3.7.0",
    "pyarrow>=14.0.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "flake8>=6.1.0",
]
docs = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=1.3.0",
]

[project.scripts]
claims-fraud = "claims_fraud.cli:main"

[project.urls]
Homepage = "https://github.com/yourusername/claims-fraud"
Documentation = "https://claims-fraud.readthedocs.io"
Repository = "https://github.com/yourusername/claims-fraud"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
claims_fraud = ["py.typed", "configs/*.yaml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=claims_fraud --cov-report=html --cov-report=term"

[tool.black]
line-length = 100
target-version = ["py39", "py310", "py311"]
include = '\\.pyi?$'

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
'''
        pyproject_path = self.base_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        print(f"✅ Created pyproject.toml: {pyproject_path}")
    
    def create_setup_cfg(self):
        """Create setup.cfg for additional configuration"""
        setup_cfg_content = '''[metadata]
name = claims-fraud
version = attr: claims_fraud.__version__.__version__
description = AI-powered fraud detection system for insurance claims
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/yourusername/claims-fraud
author = Claims Fraud Detection Team
license = MIT
classifiers =
    Development Status :: 4 - Beta
    Intended Audience :: Developers
    License :: OSI Approved :: MIT License
    Programming Language :: Python :: 3.9
    Programming Language :: Python :: 3.10
    Programming Language :: Python :: 3.11

[options]
package_dir =
    = src
packages = find:
python_requires = >=3.9
install_requires =
    numpy>=1.24.0,<2.0
    pandas>=2.0.0
    # ... (other dependencies)

[options.packages.find]
where = src

[options.entry_points]
console_scripts =
    claims-fraud = claims_fraud.cli:main
'''
        setup_cfg_path = self.base_path / "setup.cfg"
        setup_cfg_path.write_text(setup_cfg_content)
        print(f"✅ Created setup.cfg: {setup_cfg_path}")
    
    def create_manifest(self):
        """Create MANIFEST.in"""
        manifest_content = '''include README.md
include LICENSE
include pyproject.toml
include setup.cfg
recursive-include src/claims_fraud/configs *.yaml
recursive-include docs *.md *.rst
recursive-exclude * __pycache__
recursive-exclude * *.py[co]
recursive-exclude * .DS_Store
'''
        manifest_path = self.base_path / "MANIFEST.in"
        manifest_path.write_text(manifest_content)
        print(f"✅ Created MANIFEST.in: {manifest_path}")
    
    def create_readme(self):
        """Create comprehensive README.md"""
        readme_content = '''# Claims Fraud Detection 🕵️

AI-powered fraud detection system for insurance claims using tree-based models, fairness analysis, and drift monitoring.

## 🚀 Quick Start

### Installation

```bash
pip install claims-fraud
```

### Basic Usage

```python
from claims_fraud import FraudScorer

# Load pre-trained model
scorer = FraudScorer.from_pretrained("path/to/model")

# Score claims
fraud_scores = scorer.predict(claims_data)

# Get detailed analysis
analysis = scorer.analyze(claims_data)
```

### CLI Usage

```bash
# Train a model
claims-fraud train --config configs/default.yaml --data data/train.parquet

# Score claims
claims-fraud score --model models/model.pkl --input data/test.parquet

# Launch web dashboard
claims-fraud serve --port 8501
```

## 📦 Features

- **Fraud Detection**: Tree-based models (XGBoost, CatBoost) for anomaly detection
- **Fairness Analysis**: Detect and mitigate bias across protected attributes
- **Drift Monitoring**: PSI-based monitoring for model degradation
- **Explainability**: SHAP values for model interpretability
- **Web Dashboard**: Interactive Streamlit dashboard for analysis
- **CLI Tools**: Command-line tools for training and scoring

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quickstart.md)
- [API Reference](docs/api/)
- [Examples](examples/)

## 🛠️ Development

```bash
# Clone repository
git clone https://github.com/yourusername/claims-fraud.git
cd claims-fraud

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/
```

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## 📧 Contact

For questions or issues, please open a GitHub issue.
'''
        readme_path = self.base_path / "README_NEW.md"
        readme_path.write_text(readme_content)
        print(f"✅ Created README: {readme_path}")
    
    def run(self):
        """Run the modularization process"""
        print("=" * 60)
        print("🎯 Claims Fraud Detection - Package Modularization")
        print("=" * 60)
        
        # Step 1: Backup
        self.create_backup()
        
        # Step 2: Create structure
        self.create_new_structure()
        
        # Step 3: Create configuration files
        self.create_version_file()
        self.create_main_init()
        self.create_pyproject_toml()
        self.create_setup_cfg()
        self.create_manifest()
        self.create_readme()
        
        print("\n" + "=" * 60)
        print("✅ Package structure created successfully!")
        print("=" * 60)
        print("\n📋 Next Steps:")
        print("1. Review the new structure in src/claims_fraud/")
        print("2. Run: python modularize_step2_migrate.py")
        print("3. Test the package: pip install -e .")
        print("4. Run tests: pytest")
        print("\n💡 Backup location:", self.backup_dir)


if __name__ == "__main__":
    import sys
    
    base_path = "/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder"
    
    modularizer = PackageModularizer(base_path)
    
    try:
        modularizer.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
