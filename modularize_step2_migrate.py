#!/usr/bin/env python3
"""
Automated Package Modularization Script
Step 2: Migrate existing code to new structure
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

class CodeMigrator:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.old_src = self.base_path / "src"
        self.new_package = self.base_path / "src" / "claims_fraud"
        
        # Define migration mapping
        self.migration_map = {
            # Core models
            "tree_models.py": "core/tree_models.py",
            "model_architecture.py": "core/models.py",
            
            # Data handling
            "data_ingestion.py": "data/ingestion.py",
            "preprocessing.py": "data/preprocessing.py",
            
            # Analysis
            "fairness_analysis.py": "analysis/fairness.py",
            "psi_monitoring.py": "analysis/monitoring.py",
            "evaluation.py": "analysis/evaluation.py",
            
            # Configuration
            "config_manager.py": "config/manager.py",
            
            # ML operations
            "training.py": "ml/training.py",
            "hyperparameter_tuning.py": "ml/tuning.py",
            "model_registry.py": "ml/registry.py",
            
            # Batch operations
            "batch_scoring.py": "core/scoring.py",
        }
    
    def copy_file_with_modifications(self, source: Path, dest: Path):
        """Copy file and update imports"""
        print(f"  📄 Migrating: {source.name} → {dest.relative_to(self.base_path)}")
        
        # Read source
        content = source.read_text()
        
        # Update imports (basic transformation)
        # from src.X import Y → from claims_fraud.X import Y
        content = content.replace("from src.", "from claims_fraud.")
        content = content.replace("import src.", "import claims_fraud.")
        
        # Write to destination
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
    
    def migrate_code_files(self):
        """Migrate Python files to new structure"""
        print("\n🔄 Migrating code files...")
        
        for old_file, new_location in self.migration_map.items():
            source = self.old_src / old_file
            dest = self.new_package / new_location
            
            if source.exists():
                self.copy_file_with_modifications(source, dest)
            else:
                print(f"  ⚠️  Not found: {old_file}")
        
        print("✅ Code migration complete!")
    
    def split_webapp(self):
        """Split monolithic webapp into components"""
        print("\n🔪 Splitting webapp into components...")
        
        webapp_path = self.old_src / "webapp_enhanced.py"
        if not webapp_path.exists():
            print("  ⚠️  webapp_enhanced.py not found")
            return
        
        content = webapp_path.read_text()
        
        # Create main app file
        app_dest = self.new_package / "ui" / "app.py"
        
        # For now, copy entire file - will be split manually later
        app_dest.write_text(content)
        print(f"  📄 Created: {app_dest.relative_to(self.base_path)}")
        print("  💡 Note: Webapp will need manual component splitting")
        
        print("✅ Webapp migration complete!")
    
    def create_cli_commands(self):
        """Create CLI command files"""
        print("\n⌨️  Creating CLI commands...")
        
        # Train command
        train_cli = '''"""Training CLI command"""
import click
from pathlib import Path
from claims_fraud.ml.training import train_model
from claims_fraud.config.manager import ConfigManager

@click.command()
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Path to configuration file')
@click.option('--data', type=click.Path(exists=True), required=True,
              help='Path to training data')
@click.option('--output', type=click.Path(), default='models/model.pkl',
              help='Output path for trained model')
def train(config, data, output):
    """Train a fraud detection model"""
    click.echo(f"Training model with config: {config}")
    
    # Load config
    config_manager = ConfigManager(config)
    cfg = config_manager.get_config()
    
    # Train model
    model = train_model(cfg, data_path=data)
    
    # Save model
    model.save(output)
    click.echo(f"Model saved to: {output}")

if __name__ == '__main__':
    train()
'''
        
        # Score command
        score_cli = '''"""Scoring CLI command"""
import click
import pandas as pd
from pathlib import Path
from claims_fraud.core.scoring import FraudScorer

@click.command()
@click.option('--model', type=click.Path(exists=True), required=True,
              help='Path to trained model')
@click.option('--input', 'input_file', type=click.Path(exists=True), required=True,
              help='Path to input data')
@click.option('--output', type=click.Path(), required=True,
              help='Path to output results')
def score(model, input_file, output):
    """Score claims for fraud"""
    click.echo(f"Scoring data: {input_file}")
    
    # Load model and data
    scorer = FraudScorer.from_file(model)
    data = pd.read_parquet(input_file)
    
    # Score
    scores = scorer.predict(data)
    
    # Save results
    results = pd.DataFrame({'fraud_score': scores})
    results.to_csv(output, index=False)
    click.echo(f"Results saved to: {output}")

if __name__ == '__main__':
    score()
'''
        
        # Serve command
        serve_cli = '''"""Web app serving CLI command"""
import click
import subprocess
from pathlib import Path

@click.command()
@click.option('--port', default=8501, help='Port to run the app on')
@click.option('--config', type=click.Path(exists=True), default='configs/default.yaml',
              help='Configuration file')
def serve(port, config):
    """Launch the Streamlit web dashboard"""
    click.echo(f"Starting dashboard on port {port}")
    
    app_path = Path(__file__).parent.parent / "ui" / "app.py"
    
    subprocess.run([
        "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--", "--config", config
    ])

if __name__ == '__main__':
    serve()
'''
        
        # Main CLI
        main_cli = '''"""Main CLI entry point"""
import click
from .train import train
from .score import score
from .serve import serve

@click.group()
@click.version_option(version='0.1.0', prog_name='claims-fraud')
def main():
    """Claims Fraud Detection CLI"""
    pass

# Register commands
main.add_command(train)
main.add_command(score)
main.add_command(serve)

if __name__ == '__main__':
    main()
'''
        
        # Write CLI files
        cli_dir = self.new_package / "cli"
        cli_dir.mkdir(parents=True, exist_ok=True)
        
        (cli_dir / "train.py").write_text(train_cli)
        (cli_dir / "score.py").write_text(score_cli)
        (cli_dir / "serve.py").write_text(serve_cli)
        (cli_dir / "__init__.py").write_text(main_cli)
        
        print("  ✅ Created train command")
        print("  ✅ Created score command")
        print("  ✅ Created serve command")
        print("  ✅ Created main CLI")
        
        print("✅ CLI creation complete!")
    
    def create_utility_modules(self):
        """Create utility modules"""
        print("\n🔧 Creating utility modules...")
        
        # Logging utility
        logging_util = '''"""Logging configuration"""
import logging
import sys
from pathlib import Path

def setup_logging(name: str, level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
'''
        
        # Path utility
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
        return self.base_path / "data"
    
    @property
    def models_dir(self) -> Path:
        """Get models directory"""
        return self.base_path / "models"
    
    @property
    def config_dir(self) -> Path:
        """Get config directory"""
        return self.base_path / "configs"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory"""
        return self.base_path / "logs"
    
    def ensure_dirs(self):
        """Ensure all directories exist"""
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
'''
        
        utils_dir = self.new_package / "utils"
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        (utils_dir / "logging.py").write_text(logging_util)
        (utils_dir / "paths.py").write_text(paths_util)
        
        print("  ✅ Created logging utility")
        print("  ✅ Created paths utility")
        
        print("✅ Utility modules created!")
    
    def create_example_scripts(self):
        """Create example usage scripts"""
        print("\n📝 Creating example scripts...")
        
        quickstart = '''"""Quickstart example for Claims Fraud Detection"""
from claims_fraud import FraudScorer, ClaimsTreeAutoencoder
import pandas as pd

def main():
    # Load sample data
    data = pd.read_parquet("data/sample_claims.parquet")
    
    # Create and train model
    model = ClaimsTreeAutoencoder(model_type="catboost")
    model.fit(data, 
              categorical_features=['claim_type', 'patient_gender'],
              numerical_features=['claim_amount', 'patient_age'])
    
    # Create scorer
    scorer = FraudScorer(model)
    
    # Score claims
    fraud_scores = scorer.predict(data)
    
    # Analyze results
    high_risk = fraud_scores > fraud_scores.quantile(0.95)
    print(f"Found {high_risk.sum()} high-risk claims")
    
    # Save model
    scorer.save("models/fraud_model.pkl")
    print("Model saved!")

if __name__ == "__main__":
    main()
'''
        
        examples_dir = self.base_path / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)
        
        (examples_dir / "quickstart.py").write_text(quickstart)
        
        print("  ✅ Created quickstart example")
        print("✅ Example scripts created!")
    
    def run(self):
        """Run the migration process"""
        print("=" * 60)
        print("🔄 Claims Fraud Detection - Code Migration")
        print("=" * 60)
        
        # Migrate core code
        self.migrate_code_files()
        
        # Split webapp
        self.split_webapp()
        
        # Create CLI
        self.create_cli_commands()
        
        # Create utilities
        self.create_utility_modules()
        
        # Create examples
        self.create_example_scripts()
        
        print("\n" + "=" * 60)
        print("✅ Code migration completed!")
        print("=" * 60)
        print("\n📋 Next Steps:")
        print("1. Review migrated code in src/claims_fraud/")
        print("2. Manually split webapp_enhanced.py into components")
        print("3. Update imports in all files")
        print("4. Test installation: pip install -e .")
        print("5. Run tests: pytest")
        print("\n💡 Check src/claims_fraud/ for migrated modules")


if __name__ == "__main__":
    import sys
    
    base_path = "/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder"
    
    migrator = CodeMigrator(base_path)
    
    try:
        migrator.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
