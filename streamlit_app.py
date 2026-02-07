"""
Fraud Detection Dashboard - Streamlit Web App

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(
    page_title="Claims Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Claims Fraud Detection Dashboard")
st.markdown("Real-time monitoring and analysis of fraud detection models")

# Sidebar
st.sidebar.header("Configuration")

# Connection settings (in production, use Databricks SQL connector)
use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)

if not use_sample_data:
    st.sidebar.text_input("Databricks Host", value="dbc-d4506e69-bbc8.cloud.databricks.com")
    st.sidebar.text_input("Access Token", type="password")
    catalog = st.sidebar.text_input("Catalog", value="workspace")
    schema = st.sidebar.text_input("Schema", value="default")
else:
    # Generate sample data for demo
    @st.cache_data
    def load_sample_data():
        # Model comparison data
        models_data = pd.DataFrame([
            {'model': 'CatBoost', 'mean_score': 209.5, 'p95_score': 380.2, 'p99_score': 450.1, 'training_time': 45},
            {'model': 'XGBoost', 'mean_score': 207.8, 'p95_score': 375.8, 'p99_score': 445.3, 'training_time': 42},
            {'model': 'FT-Transformer', 'mean_score': 211.2, 'p95_score': 385.5, 'p99_score': 455.7, 'training_time': 120}
        ])
        
        # Fairness data
        fairness_data = pd.DataFrame([
            {'attribute': 'Gender', 'group': 'Male', 'flagged_rate': 0.052, 'disparate_impact': 0.95, 'is_fair': True},
            {'attribute': 'Gender', 'group': 'Female', 'flagged_rate': 0.055, 'disparate_impact': 1.0, 'is_fair': True},
            {'attribute': 'Age Group', 'group': '18-35', 'flagged_rate': 0.048, 'disparate_impact': 0.88, 'is_fair': True},
            {'attribute': 'Age Group', 'group': '36-50', 'flagged_rate': 0.053, 'disparate_impact': 0.97, 'is_fair': True},
            {'attribute': 'Age Group', 'group': '51+', 'flagged_rate': 0.055, 'disparate_impact': 1.0, 'is_fair': True},
            {'attribute': 'Region', 'group': 'Northeast', 'flagged_rate': 0.048, 'disparate_impact': 0.88, 'is_fair': True},
            {'attribute': 'Region', 'group': 'South', 'flagged_rate': 0.056, 'disparate_impact': 1.02, 'is_fair': True},
        ])
        
        # PSI data
        psi_data = pd.DataFrame([
            {'feature': 'claim_amount', 'psi_score': 0.05, 'drift_severity': 'low'},
            {'feature': 'patient_age', 'psi_score': 0.08, 'drift_severity': 'low'},
            {'feature': 'claim_type', 'psi_score': 0.12, 'drift_severity': 'medium'},
            {'feature': 'provider_specialty', 'psi_score': 0.06, 'drift_severity': 'low'},
        ])
        
        # Time series data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        timeseries_data = pd.DataFrame({
            'date': dates,
            'claims_processed': np.random.randint(800, 1200, 30),
            'high_risk_flags': np.random.randint(40, 80, 30),
            'avg_fraud_score': np.random.uniform(200, 220, 30)
        })
        
        return models_data, fairness_data, psi_data, timeseries_data
    
    models_df, fairness_df, psi_df, timeseries_df = load_sample_data()

# Main dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "🤖 Model Performance", 
    "⚖️ Fairness Analysis", 
    "📈 Drift Monitoring",
    "🔍 Score Claims"
])

# Tab 1: Overview
with tab1:
    st.header("Dashboard Overview")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Best Model",
            "FT-Transformer",
            "211.2 avg score"
        )
    
    with col2:
        st.metric(
            "Claims Processed Today",
            "1,047",
            "+12% vs yesterday"
        )
    
    with col3:
        st.metric(
            "High-Risk Flags",
            "62",
            "-3 vs yesterday"
        )
    
    with col4:
        st.metric(
            "Fairness Status",
            "✓ PASS",
            "All groups fair"
        )
    
    # Time series chart
    st.subheader("Claims Processing Trend (Last 30 Days)")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeseries_df['date'],
        y=timeseries_df['claims_processed'],
        name='Claims Processed',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=timeseries_df['date'],
        y=timeseries_df['high_risk_flags'],
        name='High-Risk Flags',
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        yaxis=dict(title='Claims Processed'),
        yaxis2=dict(title='High-Risk Flags', overlaying='y', side='right'),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Model Performance
with tab2:
    st.header("Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart: Mean scores
        fig = px.bar(
            models_df,
            x='model',
            y='mean_score',
            title='Mean Fraud Score by Model',
            color='mean_score',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart: Training time
        fig = px.bar(
            models_df,
            x='model',
            y='training_time',
            title='Training Time (minutes)',
            color='training_time',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.subheader("Detailed Model Metrics")
    st.dataframe(
        models_df.style.highlight_max(subset=['mean_score'], color='lightgreen'),
        use_container_width=True
    )
    
    # Model recommendation
    best_model = models_df.loc[models_df['mean_score'].idxmax(), 'model']
    st.success(f"🏆 Recommended Model: **{best_model}** (Highest mean fraud score)")

# Tab 3: Fairness Analysis
with tab3:
    st.header("Fairness Analysis")
    
    # Fairness summary
    bias_detected = not fairness_df['is_fair'].all()
    
    if bias_detected:
        st.error("⚠️ BIAS DETECTED - Review required")
    else:
        st.success("✅ NO BIAS DETECTED - Model is fair across all groups")
    
    # Fairness by attribute
    st.subheader("Disparate Impact by Protected Attribute")
    
    fig = px.bar(
        fairness_df,
        x='group',
        y='disparate_impact',
        color='attribute',
        title='Disparate Impact Ratio (Target: 0.8-1.25)',
        barmode='group'
    )
    
    # Add reference lines
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Min Threshold")
    fig.add_hline(y=1.25, line_dash="dash", line_color="red", annotation_text="Max Threshold")
    fig.add_hline(y=1.0, line_dash="dot", line_color="green", annotation_text="Perfect Parity")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed fairness table
    st.subheader("Fairness Metrics Detail")
    st.dataframe(
        fairness_df.style.apply(
            lambda x: ['background-color: lightcoral' if not v else '' for v in x],
            subset=['is_fair']
        ),
        use_container_width=True
    )

# Tab 4: Drift Monitoring
with tab4:
    st.header("Data Drift Monitoring (PSI)")
    
    # PSI summary
    max_psi = psi_df['psi_score'].max()
    
    if max_psi >= 0.2:
        st.error("🚨 MAJOR DRIFT DETECTED - Model retraining recommended")
    elif max_psi >= 0.1:
        st.warning("⚠️ MEDIUM DRIFT DETECTED - Monitor closely")
    else:
        st.success("✅ NO SIGNIFICANT DRIFT - Data distribution is stable")
    
    # PSI scores chart
    fig = px.bar(
        psi_df,
        x='feature',
        y='psi_score',
        color='drift_severity',
        title='Population Stability Index by Feature',
        color_discrete_map={'low': 'green', 'medium': 'orange', 'high': 'red'}
    )
    
    # Add reference lines
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Medium Drift")
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Major Drift")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # PSI interpretation
    st.info("""
    **PSI Interpretation:**
    - PSI < 0.1: No significant drift
    - 0.1 ≤ PSI < 0.2: Medium drift - monitor closely
    - PSI ≥ 0.2: Major drift - consider retraining
    """)
    
    # Detailed PSI table
    st.subheader("PSI Scores Detail")
    st.dataframe(psi_df, use_container_width=True)

# Tab 5: Score Claims
with tab5:
    st.header("Score Individual Claims")
    
    st.markdown("Enter claim details to get fraud risk score")
    
    col1, col2 = st.columns(2)
    
    with col1:
        claim_amount = st.number_input("Claim Amount ($)", min_value=0, value=5000, step=100)
        patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
        claim_type = st.selectbox("Claim Type", ["Medical", "Surgical", "Emergency", "Preventive"])
    
    with col2:
        provider_specialty = st.selectbox(
            "Provider Specialty",
            ["Cardiology", "Orthopedics", "Neurology", "General Practice", "Oncology"]
        )
        geographic_region = st.selectbox(
            "Geographic Region",
            ["Northeast", "South", "Midwest", "West"]
        )
        model_choice = st.selectbox("Model", ["FT-Transformer", "CatBoost", "XGBoost"])
    
    if st.button("Score Claim", type="primary"):
        # Simulate scoring
        base_score = claim_amount / 25
        age_factor = (patient_age - 45) / 10
        type_factor = {"Medical": 0, "Surgical": 20, "Emergency": 30, "Preventive": -10}[claim_type]
        
        fraud_score = base_score + age_factor + type_factor + np.random.normal(0, 20)
        
        # Determine risk level
        if fraud_score > 400:
            risk_level = "🔴 HIGH RISK"
            risk_color = "red"
        elif fraud_score > 300:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "orange"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "green"
        
        # Display results
        st.markdown("---")
        st.subheader("Fraud Risk Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fraud Score", f"{fraud_score:.1f}")
        
        with col2:
            st.metric("Risk Level", risk_level)
        
        with col3:
            st.metric("Model Used", model_choice)
        
        # Recommendation
        if fraud_score > 400:
            st.error("⚠️ **Action Required:** This claim should be flagged for manual review")
        elif fraud_score > 300:
            st.warning("⚠️ **Caution:** Consider additional verification")
        else:
            st.success("✅ **Low Risk:** Claim can be processed normally")

# Footer
st.markdown("---")
st.markdown("Dashboard updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
