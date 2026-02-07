"""
Launcher script for Streamlit dashboard
This script sets up the proper Python path and runs the app
"""
import sys
from pathlib import Path

# Add parent directory to path so relative imports work
app_dir = Path(__file__).parent
package_dir = app_dir.parent
sys.path.insert(0, str(package_dir))

# Now import and run the actual app
if __name__ == "__main__":
    # Import with absolute path now that sys.path is set
    from claims_fraud.ui import app
    
    # Streamlit will automatically run the main() function
    app.main()
