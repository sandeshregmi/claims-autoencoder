"""
Hyperparameter tuning script entry point with proper imports
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run
from src.hyperparameter_tuning import tune_and_train

if __name__ == "__main__":
    tune_and_train()
