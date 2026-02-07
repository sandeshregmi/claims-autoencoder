"""
Launch Enhanced Fraud Detection Web App
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run enhanced webapp
from src.webapp_enhanced import main

if __name__ == "__main__":
    main()
