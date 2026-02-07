#!/bin/bash

# Apply Study Period and Data Pull Date patch to webapp_enhanced.py

FILE="/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_enhanced.py"

# Create backup
cp "$FILE" "${FILE}.backup_$(date +%Y%m%d_%H%M%S)"

# Apply the patch using sed
sed -i.tmp '/st.markdown("**Real-time fraud detection powered by AI**")/a\
    \
    # Study Period and Data Pull Date\
    col1, col2, col3, col4 = st.columns([2, 2, 2, 3])\
    \
    with col1:\
        study_period_start = st.date_input(\
            "📅 Study Period Start",\
            value=pd.to_datetime("2024-01-01"),\
            help="Start date of the study period"\
        )\
    \
    with col2:\
        study_period_end = st.date_input(\
            "📅 Study Period End",\
            value=pd.to_datetime("2024-12-31"),\
            help="End date of the study period"\
        )\
    \
    with col3:\
        data_pull_date = st.date_input(\
            "📊 Data Pull Date",\
            value=pd.to_datetime("today"),\
            help="Date when the data was extracted"\
        )\
    \
    with col4:\
        # Display study period summary\
        if study_period_start and study_period_end:\
            if study_period_end >= study_period_start:\
                duration_days = (study_period_end - study_period_start).days + 1\
                duration_months = duration_days / 30.44\
                st.info(f"**Duration:** {duration_days} days ({duration_months:.1f} months)")\
            else:\
                st.error("⚠️ End date must be after start date!")\
    \
    # Store in session state for use across tabs\
    st.session_state.study_period_start = study_period_start\
    st.session_state.study_period_end = study_period_end\
    st.session_state.data_pull_date = data_pull_date
' "$FILE"

# Remove temporary file
rm -f "${FILE}.tmp"

echo "✅ Patch applied successfully!"
echo "📁 Backup saved to: ${FILE}.backup_*"
echo ""
echo "The study period and data pull date fields have been added to the top of the dashboard."
echo ""
echo "To see the changes, restart your application:"
echo "  ./start.sh"
