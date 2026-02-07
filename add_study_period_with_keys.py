#!/usr/bin/env python3
"""
Add Study Period and Data Pull Date to webapp_enhanced.py
WITH unique keys to avoid duplicate ID errors
"""

import sys
from pathlib import Path
import re
from datetime import datetime

def add_study_period_fields(file_path):
    """Add study period and data pull date fields to the webapp."""
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Backup created: {backup_path}")
    
    # The code to insert - WITH unique keys
    study_period_code = '''
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
    
    # Store in session state for use across tabs
    st.session_state.study_period_start = study_period_start
    st.session_state.study_period_end = study_period_end
    st.session_state.data_pull_date = data_pull_date
    '''
    
    # Find the location - look for the exact pattern in the file
    # Pattern: after the "Real-time fraud detection..." line and before the next st.markdown("---")
    pattern = r'(st\.markdown\(".*Real-time fraud detection.*"\)\s*\n\s*st\.markdown\("---"\))'
    
    # Check if pattern exists
    if not re.search(pattern, content):
        print("Trying alternative pattern...")
        
        # Try alternative pattern - simpler
        pattern = r'(st\.markdown\("---"\))'
        matches = list(re.finditer(pattern, content))
        
        if len(matches) < 1:
            print("❌ Could not find any --- separator in file")
            return False
        
        # Find the first --- after the header
        # Look for the one that comes after "Real-time fraud detection"
        header_pos = content.find('Real-time fraud detection')
        if header_pos == -1:
            print("❌ Could not find 'Real-time fraud detection' text")
            return False
        
        # Find the first --- after the header
        target_match = None
        for match in matches:
            if match.start() > header_pos:
                target_match = match
                break
        
        if not target_match:
            print("❌ Could not find --- separator after header")
            return False
        
        # Insert before the ---
        insert_pos = target_match.start()
        new_content = content[:insert_pos] + study_period_code + '\n    ' + content[insert_pos:]
    else:
        # Use the original pattern
        new_content = re.sub(
            pattern,
            lambda m: m.group(1)[:-len('st.markdown("---")')] + study_period_code + '\n    st.markdown("---")',
            content
        )
    
    # Write the modified content
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Study period and data pull date fields added successfully!")
    print("")
    print("Changes made:")
    print("  - Added 3 date input fields with unique keys")
    print("  - Stored dates in session state")
    print("")
    print("To see the changes, restart your application:")
    print("  ./start.sh")
    
    return True


if __name__ == "__main__":
    file_path = "/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_enhanced.py"
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    success = add_study_period_fields(file_path)
    sys.exit(0 if success else 1)
