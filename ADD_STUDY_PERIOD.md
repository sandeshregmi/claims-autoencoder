# Add Study Period and Data Pull Date to Dashboard

## What to Add

Add "Study Period" and "Data Pull Date" input fields at the top of the dashboard, right after the main header.

## Location

File: `src/webapp_enhanced.py`

Find this section (around line 850-860):

```python
def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">🚨 Claims Fraud Detection Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("**Real-time fraud detection powered by AI**")
    st.markdown("---")
```

## Code to Add

Replace the above section with:

```python
def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">🚨 Claims Fraud Detection Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("**Real-time fraud detection powered by AI**")
    
    # Study Period and Data Pull Date at the top
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
            duration_days = (study_period_end - study_period_start).days
            st.info(f"**Study Duration:** {duration_days} days ({duration_days/30:.1f} months)")
    
    st.markdown("---")
```

## Visual Result

After adding this code, the top of your dashboard will look like:

```
===========================================
🚨 Claims Fraud Detection Dashboard
===========================================
Real-time fraud detection powered by AI

📅 Study Period Start  | 📅 Study Period End  | 📊 Data Pull Date     | ℹ️ Study Duration: 365 days (12.2 months)
[Date Picker]          | [Date Picker]        | [Date Picker]         |
-------------------------------------------
```

## Features

1. **Study Period Start**: Select the beginning date of your analysis period
2. **Study Period End**: Select the end date of your analysis period  
3. **Data Pull Date**: When the data was extracted from the source system
4. **Study Duration**: Automatically calculated display showing days and months

## Default Values

- Study Period Start: January 1, 2024
- Study Period End: December 31, 2024  
- Data Pull Date: Today's date (automatically updated)

## How to Use the Values

These date inputs are stored in variables and can be used later in your code. For example:

```python
# Filter data based on study period
filtered_data = st.session_state.data[
    (st.session_state.data['claim_date'] >= study_period_start) &
    (st.session_state.data['claim_date'] <= study_period_end)
]

# Display in reports
st.write(f"Analysis Period: {study_period_start} to {study_period_end}")
st.write(f"Data as of: {data_pull_date}")
```

## Customization Options

### Change Default Dates
```python
study_period_start = st.date_input(
    "📅 Study Period Start",
    value=pd.to_datetime("2023-06-01"),  # Change this
    help="Start date of the study period"
)
```

### Add Min/Max Date Constraints
```python
study_period_start = st.date_input(
    "📅 Study Period Start",
    value=pd.to_datetime("2024-01-01"),
    min_value=pd.to_datetime("2020-01-01"),
    max_value=pd.to_datetime("today"),
    help="Start date of the study period"
)
```

### Store in Session State
```python
# Store dates in session state for use across tabs
if 'study_period_start' not in st.session_state:
    st.session_state.study_period_start = study_period_start
if 'study_period_end' not in st.session_state:
    st.session_state.study_period_end = study_period_end
if 'data_pull_date' not in st.session_state:
    st.session_state.data_pull_date = data_pull_date
```

## Complete Implementation

Here's the full section with proper formatting:

```python
def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">🚨 Claims Fraud Detection Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("**Real-time fraud detection powered by AI**")
    
    # ========================================
    # Study Period and Data Pull Date
    # ========================================
    st.markdown("### 📆 Analysis Period Configuration")
    
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
                st.error("End date must be after start date!")
    
    # Store in session state for use across tabs
    st.session_state.study_period_start = study_period_start
    st.session_state.study_period_end = study_period_end
    st.session_state.data_pull_date = data_pull_date
    
    st.markdown("---")
    
    # Rest of your application code continues here...
```

## Testing

After adding the code:

1. Stop the dashboard (Ctrl+C)
2. Restart: `./start.sh`
3. Check that the date pickers appear at the top
4. Try changing dates and verify the duration updates
5. Verify dates persist when switching between tabs

## Troubleshooting

**Error: "NameError: name 'pd' is not defined"**
- Make sure `import pandas as pd` is at the top of the file

**Dates don't persist across tabs**
- Add the session state storage code shown above

**Layout looks wrong**
- Adjust the column widths in `st.columns([2, 2, 2, 3])` as needed

---

**This will add professional date configuration at the top of your dashboard!** 📅
