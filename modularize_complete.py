#!/usr/bin/env python3
"""
Complete Modularization Script for Claims Fraud Detection
Executes all phases of the modularization process

Usage:
    python modularize_complete.py [--phase PHASE_NUMBER] [--dry-run]
    
Examples:
    python modularize_complete.py                    # Run all phases
    python modularize_complete.py --phase 2          # Run only phase 2
    python modularize_complete.py --dry-run          # Preview changes
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import re

class Color:
    """Terminal colors"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

class Modularizer:
    def __init__(self, base_path: str, dry_run: bool = False):
        self.base_path = Path(base_path)
        self.dry_run = dry_run
        self.new_package_name = "claims_fraud"
        self.backup_dir = None
        
        # Track progress
        self.progress = {
            'files_created': 0,
            'files_migrated': 0,
            'files_modified': 0,
            'errors': []
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with color"""
        colors = {
            "INFO": Color.CYAN,
            "SUCCESS": Color.GREEN,
            "WARNING": Color.YELLOW,
            "ERROR": Color.RED,
            "HEADER": Color.HEADER
        }
        color = colors.get(level, "")
        print(f"{color}{message}{Color.END}")
    
    def create_backup(self) -> Path:
        """Phase 0: Create backup"""
        self.log("\n" + "="*70, "HEADER")
        self.log("PHASE 0: Creating Backup", "HEADER")
        self.log("="*70, "HEADER")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_dir = self.base_path / f"_modularization_backup_{timestamp}"
        
        if self.dry_run:
            self.log(f"[DRY RUN] Would create backup at: {self.backup_dir}", "INFO")
            return self.backup_dir
        
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup critical files
        critical_files = [
            "src/",
            "setup.py",
            "requirements.txt",
            "requirements_clean.txt"
        ]
        
        for item in critical_files:
            src_path = self.base_path / item
            if src_path.exists():
                if src_path.is_dir():
                    dst_path = self.backup_dir / item
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    self.log(f"✅ Backed up directory: {item}", "SUCCESS")
                else:
                    dst_path = self.backup_dir / item
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    self.log(f"✅ Backed up file: {item}", "SUCCESS")
        
        self.log(f"\n📦 Backup location: {self.backup_dir}", "SUCCESS")
        return self.backup_dir
    
    def phase1_create_structure(self):
        """Phase 1: Create new package structure"""
        self.log("\n" + "="*70, "HEADER")
        self.log("PHASE 1: Creating Package Structure", "HEADER")
        self.log("="*70, "HEADER")
        
        # Define structure
        structure = {
            f"src/{self.new_package_name}": ["__init__.py", "__version__.py"],
            f"src/{self.new_package_name}/core": ["__init__.py", "base.py", "tree_models.py", "scoring.py", "explainability.py"],
            f"src/{self.new_package_name}/data": ["__init__.py", "ingestion.py", "preprocessing.py", "validation.py", "schemas.py"],
            f"src/{self.new_package_name}/analysis": ["__init__.py", "fairness.py", "monitoring.py", "evaluation.py"],
            f"src/{self.new_package_name}/ml": ["__init__.py", "training.py", "tuning.py", "registry.py"],
            f"src/{self.new_package_name}/config": ["__init__.py", "manager.py", "schemas.py"],
            f"src/{self.new_package_name}/ui": ["__init__.py", "app.py"],
            f"src/{self.new_package_name}/ui/components": [
                "__init__.py", "header.py", "sidebar.py", "dashboard_tab.py",
                "fraud_tab.py", "importance_tab.py", "analysis_tab.py",
                "shap_tab.py", "monitoring_tab.py", "fairness_tab.py", "export_tab.py"
            ],
            f"src/{self.new_package_name}/ui/utils": ["__init__.py", "plots.py", "formatters.py", "state.py"],
            f"src/{self.new_package_name}/utils": ["__init__.py", "logging.py", "paths.py", "decorators.py", "io.py"],
            f"src/{self.new_package_name}/cli": ["__init__.py", "train.py", "score.py", "evaluate.py", "serve.py"],
            "configs": ["default.yaml", "development.yaml", "production.yaml"],
            "tests": ["__init__.py", "conftest.py"],
            "tests/test_core": ["__init__.py", "test_models.py"],
            "tests/test_data": ["__init__.py"],
            "tests/test_analysis": ["__init__.py"],
            "tests/test_cli": ["__init__.py"],
            "examples": ["quickstart.py", "batch_scoring.py"],
            "docs": ["installation.md", "quickstart.md", "api_reference.md"],
            "scripts": ["setup_dev.sh", "run_tests.sh", "build_package.sh"],
        }
        
        for directory, files in structure.items():
            dir_path = self.base_path / directory
            
            if self.dry_run:
                self.log(f"[DRY RUN] Would create: {directory}/", "INFO")
            else:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.log(f"📁 Created: {directory}/", "INFO")
            
            for file in files:
                file_path = dir_path / file
                if not self.dry_run and not file_path.exists():
                    file_path.touch()
                    self.progress['files_created'] += 1
                self.log(f"  📄 {file}", "INFO")
        
        self.log(f"\n✅ Created {self.progress['files_created']} files", "SUCCESS")
    
    def phase2_create_core_files(self):
        """Phase 2: Create core package files"""
        self.log("\n" + "="*70, "HEADER")
        self.log("PHASE 2: Creating Core Package Files", "HEADER")
        self.log("="*70, "HEADER")
        
        # __version__.py
        version_content = '''"""Version information for claims_fraud package"""

__version__ = "0.1.0"
__author__ = "Claims Fraud Detection Team"
__license__ = "MIT"
'''
        self._write_file(f"src/{self.new_package_name}/__version__.py", version_content)
        
        # Main __init__.py
        init_content = '''"""
Claims Fraud Detection Package

A comprehensive fraud detection system for insurance claims using
tree-based models, fairness analysis, and drift monitoring.
"""

from .__version__ import __version__, __author__, __license__

# Core components (lazy imports to avoid circular dependencies)
def __getattr__(name):
    """Lazy import for better startup time"""
    if name == "FraudDetector":
        from .core.scoring import FraudDetector
        return FraudDetector
    elif name == "TreeModel":
        from .core.tree_models import ClaimsTreeAutoencoder
        return ClaimsTreeAutoencoder
    elif name == "FairnessAnalyzer":
        from .analysis.fairness import FairnessAnalyzer
        return FairnessAnalyzer
    elif name == "PSIMonitor":
        from .analysis.monitoring import PSIMonitor
        return PSIMonitor
    elif name == "DataPipeline":
        from .data.ingestion import DataIngestion
        return DataIngestion
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    '__version__',
    '__author__',
    '__license__',
    'FraudDetector',
    'TreeModel',
    'FairnessAnalyzer',
    'PSIMonitor',
    'DataPipeline',
]
'''
        self._write_file(f"src/{self.new_package_name}/__init__.py", init_content)
        
        # pyproject.toml
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
authors = [{name = "Claims Fraud Detection Team"}]

keywords = ["fraud-detection", "insurance", "machine-learning", "anomaly-detection"]

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
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
]

[project.scripts]
claims-fraud = "claims_fraud.cli:main"

[project.urls]
Homepage = "https://github.com/yourusername/claims-fraud"
Repository = "https://github.com/yourusername/claims-fraud"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=claims_fraud --cov-report=html"

[tool.black]
line-length = 100
target-version = ["py39"]

[tool.isort]
profile = "black"
line_length = 100
'''
        self._write_file("pyproject.toml", pyproject_content)
        
        # MANIFEST.in
        manifest_content = '''include README.md
include LICENSE
include pyproject.toml
recursive-include src/claims_fraud/config *.yaml
recursive-include docs *.md
recursive-exclude * __pycache__
recursive-exclude * *.py[co]
'''
        self._write_file("MANIFEST.in", manifest_content)
        
        self.log("✅ Core package files created", "SUCCESS")
    
    def phase3_migrate_modules(self):
        """Phase 3: Migrate existing modules"""
        self.log("\n" + "="*70, "HEADER")
        self.log("PHASE 3: Migrating Existing Modules", "HEADER")
        self.log("="*70, "HEADER")
        
        migration_map = {
            # Core
            "src/tree_models.py": f"src/{self.new_package_name}/core/tree_models.py",
            "src/model_architecture.py": f"src/{self.new_package_name}/core/base.py",
            "src/batch_scoring.py": f"src/{self.new_package_name}/core/scoring.py",
            
            # Data
            "src/data_ingestion.py": f"src/{self.new_package_name}/data/ingestion.py",
            "src/preprocessing.py": f"src/{self.new_package_name}/data/preprocessing.py",
            
            # Analysis
            "src/fairness_analysis.py": f"src/{self.new_package_name}/analysis/fairness.py",
            "src/psi_monitoring.py": f"src/{self.new_package_name}/analysis/monitoring.py",
            "src/evaluation.py": f"src/{self.new_package_name}/analysis/evaluation.py",
            
            # ML
            "src/training.py": f"src/{self.new_package_name}/ml/training.py",
            "src/hyperparameter_tuning.py": f"src/{self.new_package_name}/ml/tuning.py",
            "src/model_registry.py": f"src/{self.new_package_name}/ml/registry.py",
            
            # Config
            "src/config_manager.py": f"src/{self.new_package_name}/config/manager.py",
        }
        
        for src_file, dst_file in migration_map.items():
            src_path = self.base_path / src_file
            dst_path = self.base_path / dst_file
            
            if src_path.exists():
                if self.dry_run:
                    self.log(f"[DRY RUN] Would migrate: {src_file} → {dst_file}", "INFO")
                else:
                    # Read and update imports
                    content = src_path.read_text()
                    content = self._update_imports(content)
                    
                    # Write to new location
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    dst_path.write_text(content)
                    
                    self.progress['files_migrated'] += 1
                    self.log(f"✅ Migrated: {src_file}", "SUCCESS")
            else:
                self.log(f"⚠️  Not found: {src_file}", "WARNING")
        
        self.log(f"\n✅ Migrated {self.progress['files_migrated']} modules", "SUCCESS")
    
    def phase4_create_cli(self):
        """Phase 4: Create CLI"""
        self.log("\n" + "="*70, "HEADER")
        self.log("PHASE 4: Creating CLI", "HEADER")
        self.log("="*70, "HEADER")
        
        # Main CLI
        cli_main = '''"""Main CLI entry point"""
import click
from .train import train
from .score import score
from .evaluate import evaluate
from .serve import serve

@click.group()
@click.version_option(version='0.1.0', prog_name='claims-fraud')
def main():
    """Claims Fraud Detection CLI
    
    A comprehensive fraud detection system for insurance claims.
    """
    pass

# Register commands
main.add_command(train)
main.add_command(score)
main.add_command(evaluate)
main.add_command(serve)

if __name__ == '__main__':
    main()
'''
        
        # Train command
        cli_train = '''"""Training CLI command"""
import click
from pathlib import Path

@click.command()
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Configuration file')
@click.option('--data', type=click.Path(exists=True), required=True,
              help='Training data (parquet/csv)')
@click.option('--output', type=click.Path(), default='models/model.pkl',
              help='Output model path')
@click.option('--model-type', type=click.Choice(['catboost', 'xgboost']), 
              default='catboost', help='Model type')
def train(config, data, output, model_type):
    """Train a fraud detection model"""
    from claims_fraud.config.manager import ConfigManager
    from claims_fraud.ml.training import train_model
    
    click.echo(f"🎓 Training {model_type} model...")
    click.echo(f"   Config: {config}")
    click.echo(f"   Data: {data}")
    
    # Load config
    config_mgr = ConfigManager(config)
    cfg = config_mgr.get_config()
    
    # Train
    model = train_model(cfg, data_path=data, model_type=model_type)
    
    # Save
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    
    click.echo(f"✅ Model saved to: {output}")
'''
        
        # Score command
        cli_score = '''"""Scoring CLI command"""
import click
import pandas as pd

@click.command()
@click.option('--model', type=click.Path(exists=True), required=True,
              help='Trained model file')
@click.option('--input', 'input_file', type=click.Path(exists=True), required=True,
              help='Input data file')
@click.option('--output', type=click.Path(), required=True,
              help='Output scores file')
@click.option('--threshold', type=float, default=0.95,
              help='Fraud threshold percentile')
def score(model, input_file, output, threshold):
    """Score claims for fraud"""
    from claims_fraud.core.scoring import FraudDetector
    
    click.echo(f"🎯 Scoring claims...")
    click.echo(f"   Model: {model}")
    click.echo(f"   Input: {input_file}")
    
    # Load
    detector = FraudDetector.load(model)
    data = pd.read_parquet(input_file)
    
    # Score
    scores = detector.predict(data)
    
    # Save
    results = pd.DataFrame({
        'fraud_score': scores,
        'is_fraud': scores > scores.quantile(threshold)
    })
    results.to_csv(output, index=False)
    
    click.echo(f"✅ Scores saved to: {output}")
    click.echo(f"   Flagged: {results['is_fraud'].sum()} / {len(results)}")
'''
        
        # Serve command
        cli_serve = '''"""Web app serving command"""
import click
import subprocess
from pathlib import Path

@click.command()
@click.option('--port', default=8501, help='Port number')
@click.option('--config', type=click.Path(), default='configs/default.yaml',
              help='Configuration file')
def serve(port, config):
    """Launch the Streamlit web dashboard"""
    click.echo(f"🚀 Starting dashboard on port {port}...")
    
    app_path = Path(__file__).parent.parent / "ui" / "app.py"
    
    if not app_path.exists():
        click.echo(f"❌ App not found: {app_path}", err=True)
        return 1
    
    subprocess.run([
        "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--", "--config", config
    ])
'''
        
        # Evaluate command
        cli_evaluate = '''"""Evaluation CLI command"""
import click

@click.command()
@click.option('--model', type=click.Path(exists=True), required=True,
              help='Trained model file')
@click.option('--test-data', type=click.Path(exists=True), required=True,
              help='Test data file')
@click.option('--output', type=click.Path(), default='evaluation_report.txt',
              help='Output report file')
def evaluate(model, test_data, output):
    """Evaluate model performance"""
    from claims_fraud.core.scoring import FraudDetector
    from claims_fraud.analysis.evaluation import evaluate_model
    import pandas as pd
    
    click.echo(f"📊 Evaluating model...")
    
    # Load
    detector = FraudDetector.load(model)
    data = pd.read_parquet(test_data)
    
    # Evaluate
    results = evaluate_model(detector, data)
    
    # Save report
    with open(output, 'w') as f:
        f.write(results)
    
    click.echo(f"✅ Report saved to: {output}")
'''
        
        cli_files = {
            f"src/{self.new_package_name}/cli/__init__.py": cli_main,
            f"src/{self.new_package_name}/cli/train.py": cli_train,
            f"src/{self.new_package_name}/cli/score.py": cli_score,
            f"src/{self.new_package_name}/cli/serve.py": cli_serve,
            f"src/{self.new_package_name}/cli/evaluate.py": cli_evaluate,
        }
        
        for path, content in cli_files.items():
            self._write_file(path, content)
        
        self.log("✅ CLI created", "SUCCESS")
    
    def phase5_create_utils(self):
        """Phase 5: Create utility modules"""
        self.log("\n" + "="*70, "HEADER")
        self.log("PHASE 5: Creating Utility Modules", "HEADER")
        self.log("="*70, "HEADER")
        
        # Logging utility
        logging_util = '''"""Logging configuration"""
import logging
import sys
from pathlib import Path

def setup_logger(name: str, level: str = "INFO", log_file: str = None):
    """Setup logging configuration
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    
    # Formatter
    fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    
    return logger
'''
        
        # Paths utility
        paths_util = '''"""Path management utilities"""
from pathlib import Path
import os

class PathManager:
    """Manage project paths"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or os.getcwd())
    
    @property
    def data_dir(self) -> Path:
        """Get data directory"""
        path = self.base_path / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def models_dir(self) -> Path:
        """Get models directory"""
        path = self.base_path / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def config_dir(self) -> Path:
        """Get config directory"""
        return self.base_path / "configs"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory"""
        path = self.base_path / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def results_dir(self) -> Path:
        """Get results directory"""
        path = self.base_path / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path
'''
        
        utils_files = {
            f"src/{self.new_package_name}/utils/logging.py": logging_util,
            f"src/{self.new_package_name}/utils/paths.py": paths_util,
        }
        
        for path, content in utils_files.items():
            self._write_file(path, content)
        
        self.log("✅ Utility modules created", "SUCCESS")
    
    def phase6_create_examples(self):
        """Phase 6: Create example scripts"""
        self.log("\n" + "="*70, "HEADER")
        self.log("PHASE 6: Creating Example Scripts", "HEADER")
        self.log("="*70, "HEADER")
        
        quickstart = '''"""Quickstart example for Claims Fraud Detection"""
from claims_fraud import TreeModel, FraudDetector
import pandas as pd

def main():
    """Run quickstart example"""
    print("🚀 Claims Fraud Detection - Quickstart")
    print("="*50)
    
    # 1. Load data
    print("\n1. Loading sample data...")
    data = pd.read_parquet("data/sample_claims.parquet")
    print(f"   Loaded {len(data)} claims")
    
    # 2. Create model
    print("\n2. Creating model...")
    model = TreeModel(model_type="catboost")
    
    # 3. Train
    print("\n3. Training model...")
    model.fit(
        data,
        categorical_features=['claim_type', 'patient_gender'],
        numerical_features=['claim_amount', 'patient_age']
    )
    print("   ✅ Training complete!")
    
    # 4. Create detector
    print("\n4. Creating fraud detector...")
    detector = FraudDetector(model)
    
    # 5. Score claims
    print("\n5. Scoring claims...")
    scores = detector.predict(data)
    
    # 6. Analyze
    print("\n6. Analyzing results...")
    threshold = scores.quantile(0.95)
    high_risk = scores > threshold
    
    print(f"   Total claims: {len(scores)}")
    print(f"   High risk: {high_risk.sum()} ({high_risk.sum()/len(scores)*100:.1f}%)")
    print(f"   Threshold: {threshold:,.0f}")
    
    # 7. Save model
    print("\n7. Saving model...")
    detector.save("models/quickstart_model.pkl")
    print("   ✅ Model saved!")
    
    print("\n✅ Quickstart complete!")
    print("\nNext steps:")
    print("  - Try: claims-fraud serve")
    print("  - See: examples/batch_scoring.py")

if __name__ == "__main__":
    main()
'''
        
        examples = {
            "examples/quickstart.py": quickstart,
        }
        
        for path, content in examples.items():
            self._write_file(path, content)
        
        self.log("✅ Example scripts created", "SUCCESS")
    
    def phase7_create_docs(self):
        """Phase 7: Create documentation"""
        self.log("\n" + "="*70, "HEADER")
        self.log("PHASE 7: Creating Documentation", "HEADER")
        self.log("="*70, "HEADER")
        
        readme = '''# Claims Fraud Detection 🕵️

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
'''
        
        installation = '''# Installation Guide

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
'''
        
        docs = {
            "README.md": readme,
            "docs/installation.md": installation,
        }
        
        for path, content in docs.items():
            self._write_file(path, content)
        
        self.log("✅ Documentation created", "SUCCESS")
    
    def _write_file(self, path: str, content: str):
        """Write file with proper handling"""
        file_path = self.base_path / path
        
        if self.dry_run:
            self.log(f"[DRY RUN] Would write: {path}", "INFO")
            return
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        self.progress['files_created'] += 1
        self.log(f"📝 Created: {path}", "INFO")
    
    def _update_imports(self, content: str) -> str:
        """Update import statements"""
        # Replace old imports with new package name
        content = re.sub(r'from src\.', f'from {self.new_package_name}.', content)
        content = re.sub(r'import src\.', f'import {self.new_package_name}.', content)
        return content
    
    def generate_summary(self):
        """Generate summary report"""
        self.log("\n" + "="*70, "HEADER")
        self.log("MODULARIZATION SUMMARY", "HEADER")
        self.log("="*70, "HEADER")
        
        self.log(f"\n📊 Statistics:", "INFO")
        self.log(f"   Files created: {self.progress['files_created']}", "INFO")
        self.log(f"   Files migrated: {self.progress['files_migrated']}", "INFO")
        self.log(f"   Files modified: {self.progress['files_modified']}", "INFO")
        
        if self.progress['errors']:
            self.log(f"\n⚠️  Errors: {len(self.progress['errors'])}", "WARNING")
            for error in self.progress['errors']:
                self.log(f"   - {error}", "ERROR")
        
        self.log(f"\n💾 Backup: {self.backup_dir}", "INFO")
        
        self.log("\n📋 Next Steps:", "INFO")
        self.log("   1. Review new structure in src/claims_fraud/", "INFO")
        self.log("   2. Install package: pip install -e .", "INFO")
        self.log("   3. Run tests: pytest", "INFO")
        self.log("   4. Try CLI: claims-fraud --help", "INFO")
        self.log("   5. Launch dashboard: claims-fraud serve", "INFO")
        
        self.log("\n✅ Modularization complete!", "SUCCESS")
    
    def run_all_phases(self):
        """Run all modularization phases"""
        try:
            self.create_backup()
            self.phase1_create_structure()
            self.phase2_create_core_files()
            self.phase3_migrate_modules()
            self.phase4_create_cli()
            self.phase5_create_utils()
            self.phase6_create_examples()
            self.phase7_create_docs()
            self.generate_summary()
            
            return 0
        except Exception as e:
            self.log(f"\n❌ Error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return 1


def main():
    parser = argparse.ArgumentParser(description="Modularize Claims Fraud Detection package")
    parser.add_argument('--phase', type=int, help='Run specific phase only (1-7)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without executing')
    parser.add_argument('--base-path', default='/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder',
                       help='Base project path')
    
    args = parser.parse_args()
    
    modularizer = Modularizer(args.base_path, dry_run=args.dry_run)
    
    if args.dry_run:
        print(f"{Color.YELLOW}{'='*70}")
        print("DRY RUN MODE - No files will be modified")
        print(f"{'='*70}{Color.END}\n")
    
    if args.phase:
        phase_methods = {
            1: modularizer.phase1_create_structure,
            2: modularizer.phase2_create_core_files,
            3: modularizer.phase3_migrate_modules,
            4: modularizer.phase4_create_cli,
            5: modularizer.phase5_create_utils,
            6: modularizer.phase6_create_examples,
            7: modularizer.phase7_create_docs,
        }
        
        if args.phase in phase_methods:
            modularizer.create_backup()
            phase_methods[args.phase]()
            modularizer.generate_summary()
        else:
            print(f"{Color.RED}Invalid phase: {args.phase}{Color.END}")
            return 1
    else:
        return modularizer.run_all_phases()


if __name__ == "__main__":
    sys.exit(main())
