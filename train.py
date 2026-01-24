"""
Training script entry point with proper imports
"""

import sys
from pathlib import Path

# Add current directory to Python path (claims-autoencoder is the project root)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run
from src.training import main

if __name__ == "__main__":
    main()
