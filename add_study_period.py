#!/usr/bin/env python3
"""
Add Study Period and Data Pull Date to webapp_enhanced.py
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
    
    # The code to insert
    study_period_code = '''
    # Study Period and Data Pull Date
    col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
    
    with col1:
        study_period_start = st.date_input(
            "📅 Study Period Start",
            value=pd.to_datetime("2024-01-01"),
            help="Start date of the study period"
        )
    
    with col2:
        study_period_end = st.date_input(
            "📅 Study Period End",
            value=pd.to_datetime("2024-12-31"),
            help="End date of the study period"
        )
    
    with col3:
        data_pull_date = st.date_input(
            "📊 Data Pull Date",
            value=pd.to_datetime("today"),
            help="Date when the data was extracted"
        )
    
    with col4:
        # Display study period summary
        if study_period_start and study_period_end:
            if study_period_end >= study_period_start:
                duration_days = (study_period_end - study_period_start).days + 1
                duration_months = duration_days / 30.44
                st.info(f"**Duration:** {duration_days} days ({duration_months:.1f} months)")
            else:
                st.error("⚠️ End date must be after start date!")
    
    # Store in session state for use across tabs
    st.session_state.study_period_start = study_period_start
    st.session_state.study_period_end = study_period_end
    st.session_state.data_pull_date = data_pull_date
    '''
    
    # Find the location to insert (after the header)
    pattern = r'(st\.markdown\("\\*\\*Real-time fraud detection powered by AI\\*\\*"\)\s*\n)'
    
    # Check if pattern exists
    if not re.search(pattern, content):
        print("❌ Could not find insertion point in file")
        return False
    
    # Insert the code
    new_content = re.sub(
        pattern,
        r'\1' + study_period_code + '\n',
        content
    )
    
    # Write the modified content
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Study period and data pull date fields added successfully!")
    print("")
    print("Changes made:")
    print("  - Added 3 date input fields at the top")
    print("  - Added duration calculator")
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
