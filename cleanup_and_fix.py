#!/usr/bin/env python3
"""
CLEAN UP and ADD Study Period correctly
This script will:
1. Remove all duplicate study period code
2. Add it back once with unique keys
"""

import sys
from pathlib import Path
from datetime import datetime

def fix_webapp(file_path):
    """Clean up duplicates and add study period code correctly."""
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Backup created: {backup_path}")
    
    # Find main() function start
    main_start = content.find('def main():')
    if main_start == -1:
        print("❌ Could not find main() function")
        return False
    
    # Find the first st.markdown("---") after the header
    header_end = content.find('st.markdown("**Real-time fraud detection powered by AI**")', main_start)
    if header_end == -1:
        print("❌ Could not find header end")
        return False
    
    # Find the FIRST separator after header
    first_separator = content.find('st.markdown("---")', header_end)
    if first_separator == -1:
        print("❌ Could not find separator")
        return False
    
    # Remove everything between header and first separator (this removes all the duplicate date inputs)
    before = content[:header_end + len('st.markdown("**Real-time fraud detection powered by AI**")')]
    after = content[first_separator:]
    
    # The clean code to insert (with unique keys, 3 columns, NO duration display)
    clean_code = '''
    
    # Study Period and Data Pull Date
    col1, col2, col3 = st.columns(3)
    
    with col1:
        study_period_start = st.date_input(
            "📅 Study Period Start",
            value=pd.to_datetime("2024-01-01"),
            help="Start date of the study period",
            key="study_period_start_date"
        )
    
    with col2:
        study_period_end = st.date_input(
            "📅 Study Period End",
            value=pd.to_datetime("2024-12-31"),
            help="End date of the study period",
            key="study_period_end_date"
        )
    
    with col3:
        data_pull_date = st.date_input(
            "📊 Data Pull Date",
            value=pd.to_datetime("today"),
            help="Date when the data was extracted",
            key="data_pull_date_input"
        )
    
    # Store in session state
    st.session_state.study_period_start = study_period_start
    st.session_state.study_period_end = study_period_end
    st.session_state.data_pull_date = data_pull_date
    
    '''
    
    # Reconstruct the file
    new_content = before + clean_code + after
    
    # Write the fixed content
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("✅ File cleaned and fixed successfully!")
    print("")
    print("Changes made:")
    print("  - Removed all duplicate date inputs")
    print("  - Added clean version with unique keys")
    print("  - 3 columns (no duration display)")
    print("")
    print("Restart your application:")
    print("  ./start.sh")
    
    return True


if __name__ == "__main__":
    file_path = "/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_enhanced.py"
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    success = fix_webapp(file_path)
    sys.exit(0 if success else 1)
