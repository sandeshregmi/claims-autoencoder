"""
Enhanced Streamlit Web Application for Claims Fraud Detection
Integrated with Tree Models and Feature Importance Visualizations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from src.config_manager import ConfigManager
from src.data_ingestion import DataIngestion
from src.tree_models import ClaimsTreeAutoencoder
from src.psi_monitoring import PSIMonitor

# Try to import SHAP
try:
    import shap
    from shap_explainer import ClaimsShapExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    ClaimsShapExplainer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Claims Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'model' not in st.session_state:
    st.session_state.model = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'fraud_scores' not in st.session_state:
    st.session_state.fraud_scores = None
if 'per_feature_errors' not in st.session_state:
    st.session_state.per_feature_errors = None
if 'shap_explainer' not in st.session_state:
    st.session_state.shap_explainer = None
if 'shap_values_cache' not in st.session_state:
    st.session_state.shap_values_cache = {}
if 'psi_monitor' not in st.session_state:
    st.session_state.psi_monitor = None
if 'psi_results' not in st.session_state:
    st.session_state.psi_results = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None


@st.cache_resource
def load_model_and_config(model_type='catboost', config_path='config/example_config.yaml'):
    """Load model and configuration."""
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        
        model = ClaimsTreeAutoencoder(model_type=model_type)
        
        return model, config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


@st.cache_data
def load_data(config):
    """Load training/validation data."""
    try:
        data_ingestion = DataIngestion(config)
        train_df, val_df, test_df = data_ingestion.load_train_val_test()
        return train_df, val_df, test_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None


def train_model_cached(model, data, cat_features, num_features):
    """Train model (cached)."""
    with st.spinner("Training model... This may take 10-15 seconds"):
        model.fit(data, cat_features, num_features, verbose=False)
    return model


def plot_fraud_distribution(fraud_scores):
    """Create fraud score distribution plot."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Fraud Score Distribution', 'Log Scale Distribution')
    )
    
    # Linear scale
    fig.add_trace(
        go.Histogram(x=fraud_scores, nbinsx=50, name='Fraud Scores',
                    marker_color='steelblue'),
        row=1, col=1
    )
    
    # Add percentile lines
    p95 = np.percentile(fraud_scores, 95)
    p99 = np.percentile(fraud_scores, 99)
    
    fig.add_vline(x=p95, line_dash="dash", line_color="orange",
                 annotation_text="95th %", row=1, col=1)
    fig.add_vline(x=p99, line_dash="dash", line_color="red",
                 annotation_text="99th %", row=1, col=1)
    
    # Log scale
    fig.add_trace(
        go.Histogram(x=np.log10(fraud_scores + 1), nbinsx=50, 
                    name='Log10(Scores)', marker_color='coral'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Fraud Score", row=1, col=1)
    fig.update_xaxes(title_text="Log10(Fraud Score)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig


def plot_feature_importance(model):
    """Create feature importance plot."""
    all_importances = model.get_feature_importance()
    
    # Average importance across all targets
    avg_importance = {}
    for target_feat, importances in all_importances.items():
        for feat, imp in importances.items():
            if feat not in avg_importance:
                avg_importance[feat] = []
            avg_importance[feat].append(imp)
    
    mean_importance = {feat: np.mean(imps) for feat, imps in avg_importance.items()}
    importance_df = pd.DataFrame([
        {'feature': feat, 'importance': imp}
        for feat, imp in mean_importance.items()
    ]).sort_values('importance', ascending=False).head(15)
    
    fig = go.Figure(go.Bar(
        x=importance_df['importance'],
        y=importance_df['feature'],
        orientation='h',
        marker_color=importance_df['importance'],
        marker_colorscale='Reds'
    ))
    
    fig.update_layout(
        title="Top 15 Features for Fraud Detection",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500
    )
    fig.update_yaxes(autorange="reversed")
    
    return fig


def plot_claim_anomalies(per_feature_errors, claim_idx, fraud_score):
    """Create anomaly plot for specific claim."""
    claim_errors = {feat: errors[claim_idx] for feat, errors in per_feature_errors.items()}
    error_df = pd.DataFrame([
        {'feature': feat, 'error': err}
        for feat, err in claim_errors.items()
    ]).sort_values('error', ascending=False).head(15)
    
    fig = go.Figure(go.Bar(
        x=error_df['error'],
        y=error_df['feature'],
        orientation='h',
        marker_color=error_df['error'],
        marker_colorscale='YlOrRd'
    ))
    
    fig.update_layout(
        title=f"Feature Anomalies - Claim with Fraud Score: {fraud_score:,.0f}",
        xaxis_title="Reconstruction Error",
        yaxis_title="Feature",
        height=500
    )
    fig.update_yaxes(autorange="reversed")
    
    return fig


def plot_top_claims_heatmap(per_feature_errors, top_indices, fraud_scores, feature_names):
    """Create heatmap of top fraudulent claims."""
    top_10_errors = []
    for idx in top_indices[:10]:
        errors = [per_feature_errors[feat][idx] for feat in feature_names]
        top_10_errors.append(errors)
    
    heatmap_data = np.array(top_10_errors).T
    
    # Normalize for better visualization
    heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-6)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_normalized,
        x=[f"#{i+1} ({fraud_scores[idx]:,.0f})" for i, idx in enumerate(top_indices[:10])],
        y=feature_names,
        colorscale='YlOrRd',
        colorbar=dict(title="Normalized<br>Error")
    ))
    
    fig.update_layout(
        title="Feature Anomalies Across Top 10 Fraudulent Claims",
        xaxis_title="Claim Rank (Fraud Score)",
        yaxis_title="Feature",
        height=600
    )
    
    return fig


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">🚨 Claims Fraud Detection Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("**Real-time fraud detection powered by AI**")
    
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
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=Fraud+AI", width=150)
        st.header("⚙️ Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            options=['catboost', 'xgboost'],
            index=0,
            help="CatBoost is more stable, XGBoost is faster"
        )
        
        config_path = st.text_input(
            "Config Path",
            value="config/example_config.yaml"
        )
        
        if st.button("🔄 Load Model", type="primary"):
            model, config = load_model_and_config(model_type, config_path)
            if model and config:
                st.session_state.model = model
                st.session_state.config = config
                st.success("✅ Model loaded!")
        
        st.markdown("---")
        
        # Data loading
        st.header("📊 Data")
        
        if st.session_state.config and st.button("📁 Load Training Data"):
            train_df, val_df, test_df = load_data(st.session_state.config)
            if val_df is not None:
                st.session_state.train_data = train_df  # Store for PSI
                st.session_state.data = val_df  # Validation data
                st.session_state.test_data = test_df  # Store for PSI
                st.success(f"✅ Loaded {len(val_df)} validation claims")
                st.info(f"📊 Training: {len(train_df)}, Test: {len(test_df)}")
        
        # Train model
        if st.session_state.model and st.session_state.data is not None:
            if st.button("🎓 Train Model"):
                cat_features = st.session_state.config.data.categorical_features
                num_features = st.session_state.config.data.numerical_features
                
                st.session_state.model = train_model_cached(
                    st.session_state.model,
                    st.session_state.data,
                    cat_features,
                    num_features
                )
                st.success("✅ Model trained!")
        
        # Score claims
        if st.session_state.model and st.session_state.data is not None:
            if len(st.session_state.model.models) > 0:
                if st.button("🎯 Compute Fraud Scores"):
                    with st.spinner("Computing fraud scores..."):
                        fraud_scores, per_feature_errors = st.session_state.model.compute_fraud_scores(
                            st.session_state.data
                        )
                        st.session_state.fraud_scores = fraud_scores
                        st.session_state.per_feature_errors = per_feature_errors
                    st.success("✅ Scores computed!")
        
        st.markdown("---")
        st.caption("💡 Tip: Load model → Load data → Train → Score")
    
    # Main content tabs - conditionally add SHAP and PSI
    tabs_list = ["📊 Dashboard", "🚨 Top Frauds", "📈 Feature Importance", "🔍 Individual Analysis"]
    if SHAP_AVAILABLE:
        tabs_list.append("🔬 SHAP Explanations")
    tabs_list.append("📊 Model Monitoring")  # Add PSI tab
    tabs_list.append("⚖️ Fairness Analysis")  # Add Fairness tab
    tabs_list.append("📁 Export")
    
    tab_objects = st.tabs(tabs_list)
    tab1, tab2, tab3, tab4 = tab_objects[0], tab_objects[1], tab_objects[2], tab_objects[3]
    
    # Handle variable tab assignment based on SHAP availability
    if SHAP_AVAILABLE:
        tab_shap = tab_objects[4]
        tab_monitoring = tab_objects[5]
        tab_fairness = tab_objects[6]
        tab_export = tab_objects[7]
    else:
        tab_monitoring = tab_objects[4]
        tab_fairness = tab_objects[5]
        tab_export = tab_objects[6]
    
    # Tab 1: Dashboard
    with tab1:
        st.header("Fraud Detection Overview")
        
        if st.session_state.fraud_scores is not None:
            fraud_scores = st.session_state.fraud_scores
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Claims",
                    f"{len(fraud_scores):,}",
                    help="Total number of claims analyzed"
                )
            
            with col2:
                mean_score = fraud_scores.mean()
                st.metric(
                    "Mean Fraud Score",
                    f"{mean_score:,.0f}",
                    help="Average fraud score across all claims"
                )
            
            with col3:
                p95 = np.percentile(fraud_scores, 95)
                flagged_95 = (fraud_scores > p95).sum()
                st.metric(
                    "High Risk (95th %)",
                    f"{flagged_95:,}",
                    f"{flagged_95/len(fraud_scores)*100:.1f}%",
                    help="Claims above 95th percentile"
                )
            
            with col4:
                p99 = np.percentile(fraud_scores, 99)
                flagged_99 = (fraud_scores > p99).sum()
                st.metric(
                    "Critical (99th %)",
                    f"{flagged_99:,}",
                    f"{flagged_99/len(fraud_scores)*100:.1f}%",
                    help="Claims above 99th percentile",
                    delta_color="inverse"
                )
            
            st.markdown("---")
            
            # Distribution plot
            st.subheader("📊 Fraud Score Distribution")
            fig = plot_fraud_distribution(fraud_scores)
            st.plotly_chart(fig, width="stretch", key="fraud_distribution_main")
            
            # Threshold selector
            st.subheader("🎚️ Set Detection Threshold")
            
            col1, col2 = st.columns(2)
            
            with col1:
                threshold_type = st.radio(
                    "Threshold Type",
                    options=['Percentile', 'Absolute Value'],
                    horizontal=True
                )
            
            with col2:
                if threshold_type == 'Percentile':
                    threshold_pct = st.slider(
                        "Percentile",
                        min_value=90.0,
                        max_value=99.9,
                        value=95.0,
                        step=0.1
                    )
                    threshold = np.percentile(fraud_scores, threshold_pct)
                else:
                    threshold = st.number_input(
                        "Absolute Threshold",
                        min_value=0.0,
                        max_value=float(fraud_scores.max()),
                        value=float(np.percentile(fraud_scores, 95)),
                        step=100000.0
                    )
            
            flagged = (fraud_scores > threshold).sum()
            st.info(f"📍 Current threshold: {threshold:,.0f} → Flags {flagged:,} claims ({flagged/len(fraud_scores)*100:.2f}%)")
            
        else:
            st.info("👈 Please load model, data, and compute fraud scores from the sidebar")
            
            st.markdown("""
            ### Getting Started
            
            1. **Load Model**: Choose XGBoost or CatBoost
            2. **Load Training Data**: Load claims dataset
            3. **Train Model**: Train on loaded data (~10-15 seconds)
            4. **Compute Scores**: Calculate fraud scores for all claims
            
            Then explore:
            - 📊 **Dashboard**: Overview and distribution
            - 🚨 **Top Frauds**: Most suspicious claims
            - 📈 **Feature Importance**: What drives fraud scores
            - 🔍 **Individual Analysis**: Deep dive into specific claims
            """)
    
    # Tab 2: Top Frauds
    with tab2:
        st.header("🚨 Most Suspicious Claims")
        
        if st.session_state.fraud_scores is not None:
            fraud_scores = st.session_state.fraud_scores
            data = st.session_state.data
            
            # Number of top claims to show
            top_k = st.slider("Number of top claims", min_value=5, max_value=50, value=10)
            
            # Get top claims
            top_indices = fraud_scores.argsort()[-top_k:][::-1]
            
            # Display as cards
            for i, idx in enumerate(top_indices[:10], 1):  # Show detailed cards for top 10
                with st.expander(f"**Rank #{i}** - Fraud Score: {fraud_scores[idx]:,.0f}", expanded=(i<=3)):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("### 💰 Claim Details")
                        claim = data.iloc[idx]
                        st.write(f"**Amount**: ${claim.get('claim_amount', 'N/A'):,.2f}")
                        st.write(f"**Type**: {claim.get('claim_type', 'MISSING')}")
                        st.write(f"**Duration**: {claim.get('claim_duration_days', 'N/A')} days")
                        
                        st.markdown("### 👤 Patient")
                        st.write(f"**Age**: {claim.get('patient_age', 'N/A')}")
                        st.write(f"**Gender**: {claim.get('patient_gender', 'N/A')}")
                        st.write(f"**Previous Claims**: {claim.get('num_previous_claims', 'N/A')}")
                        
                        # Fraud indicators
                        st.markdown("### 🔍 Red Flags")
                        if pd.isna(claim.get('claim_type')):
                            st.error("⚠️ Missing claim type")
                        if claim.get('claim_amount', 0) > 10000:
                            st.warning(f"⚠️ High amount: ${claim.get('claim_amount'):,.2f}")
                        if claim.get('num_previous_claims', 0) > 3:
                            st.warning(f"⚠️ Frequent claimant: {claim.get('num_previous_claims')}")
                    
                    with col2:
                        st.markdown("### 📊 Feature Anomalies")
                        fig = plot_claim_anomalies(
                            st.session_state.per_feature_errors,
                            idx,
                            fraud_scores[idx]
                        )
                        st.plotly_chart(fig, width="stretch", key=f"anomaly_rank_{i}")
            
            # Full table
            st.markdown("---")
            st.subheader(f"📋 Top {top_k} Claims - Full Details")
            
            top_df = data.iloc[top_indices].copy()
            top_df['fraud_score'] = fraud_scores[top_indices]
            top_df['rank'] = range(1, top_k + 1)
            
            # Reorder columns
            priority_cols = ['rank', 'fraud_score', 'claim_amount', 'claim_type', 
                           'num_previous_claims', 'patient_age']
            other_cols = [c for c in top_df.columns if c not in priority_cols]
            top_df = top_df[priority_cols + other_cols]
            
            st.dataframe(top_df)
            
        else:
            st.info("Please compute fraud scores first (Dashboard tab)")
    
    # Tab 3: Feature Importance
    with tab3:
        st.header("📈 Feature Importance Analysis")
        
        if st.session_state.model and len(st.session_state.model.models) > 0:
            # Overall importance
            st.subheader("🎯 Overall Feature Importance")
            fig = plot_feature_importance(st.session_state.model)
            st.plotly_chart(fig, width="stretch", key="feature_importance_main")
            
            st.markdown("""
            **Interpretation:**
            - Higher values = more important for detecting fraud
            - These features have the strongest predictive power
            - Focus investigations on anomalies in top features
            """)
            
            # Heatmap
            if st.session_state.fraud_scores is not None:
                st.subheader("🔥 Top Claims Heatmap")
                st.markdown("Shows which features are most anomalous across the top 10 fraudulent claims")
                
                top_indices = st.session_state.fraud_scores.argsort()[-10:][::-1]
                fig = plot_top_claims_heatmap(
                    st.session_state.per_feature_errors,
                    top_indices,
                    st.session_state.fraud_scores,
                    st.session_state.model.feature_names
                )
                st.plotly_chart(fig, width="stretch", key="heatmap_top10")
        else:
            st.info("Please train the model first")
    
    # Tab 4: Individual Analysis
    with tab4:
        st.header("🔍 Individual Claim Analysis")
        
        if st.session_state.data is not None and st.session_state.fraud_scores is not None:
            # Claim selector
            claim_idx = st.number_input(
                "Enter Claim Index",
                min_value=0,
                max_value=len(st.session_state.data) - 1,
                value=int(st.session_state.fraud_scores.argmax()),
                help="Index of the claim to analyze"
            )
            
            claim = st.session_state.data.iloc[claim_idx]
            fraud_score = st.session_state.fraud_scores[claim_idx]
            
            # Display claim info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fraud Score", f"{fraud_score:,.0f}")
            
            with col2:
                percentile = (st.session_state.fraud_scores < fraud_score).sum() / len(st.session_state.fraud_scores) * 100
                st.metric("Percentile", f"{percentile:.1f}%")
            
            with col3:
                if percentile >= 99:
                    risk_level = "🔴 CRITICAL"
                elif percentile >= 95:
                    risk_level = "🟡 HIGH"
                else:
                    risk_level = "🟢 MEDIUM"
                st.metric("Risk Level", risk_level)
            
            st.markdown("---")
            
            # Claim details
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Claim Information")
                info_df = pd.DataFrame({
                    'Field': ['Amount', 'Type', 'Duration', 'Region', 'Diagnosis', 'Procedure'],
                    'Value': [
                        f"${claim.get('claim_amount', 'N/A'):,.2f}",
                        claim.get('claim_type', 'N/A'),
                        f"{claim.get('claim_duration_days', 'N/A')} days",
                        claim.get('geographic_region', 'N/A'),
                        claim.get('diagnosis_code', 'N/A'),
                        claim.get('procedure_code', 'N/A')
                    ]
                })
                # Fix PyArrow type conversion error by ensuring Value column is string type
                info_df['Value'] = info_df['Value'].astype(str)
                st.table(info_df)
            
            with col2:
                st.subheader("👤 Patient Information")
                patient_df = pd.DataFrame({
                    'Field': ['Age', 'Gender', 'Previous Claims', 'Avg Claim Amount', 'Days Since Last'],
                    'Value': [
                        claim.get('patient_age', 'N/A'),
                        claim.get('patient_gender', 'N/A'),
                        claim.get('num_previous_claims', 'N/A'),
                        f"${claim.get('average_claim_amount', 'N/A'):,.2f}" if pd.notna(claim.get('average_claim_amount')) else 'N/A',
                        f"{claim.get('days_since_last_claim', 'N/A')} days" if pd.notna(claim.get('days_since_last_claim')) else 'N/A'
                    ]
                })
                # Fix PyArrow type conversion error by ensuring Value column is string type
                patient_df['Value'] = patient_df['Value'].astype(str)
                st.table(patient_df)
            
            # Feature anomalies
            st.subheader("📊 Feature Anomaly Analysis")
            fig = plot_claim_anomalies(
                st.session_state.per_feature_errors,
                claim_idx,
                fraud_score
            )
            st.plotly_chart(fig, width="stretch", key="individual_analysis_chart")
            
            # Recommendations
            st.subheader("💡 Recommended Actions")
            
            if percentile >= 99:
                st.error("""
                **🔴 IMMEDIATE INVESTIGATION REQUIRED**
                - Suspend payment pending review
                - Request additional documentation
                - Contact provider for verification
                - Escalate to fraud investigation team
                """)
            elif percentile >= 95:
                st.warning("""
                **🟡 DETAILED REVIEW RECOMMENDED**
                - Flag for manual review
                - Verify documentation completeness
                - Check provider history
                - Monitor for patterns
                """)
            else:
                st.info("""
                **🟢 STANDARD REVIEW**
                - Process normally
                - Log for future reference
                - No immediate action required
                """)
        else:
            st.info("Please load data and compute scores first")
    

    # Tab SHAP: SHAP Explanations
    if SHAP_AVAILABLE:
        with tab_shap:
            st.header("🔬 SHAP Explanations")
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** - Understand why the model made specific predictions.
            """)
            
            if st.session_state.model is None or st.session_state.data is None:
                st.info("👈 Load model and data first")
            elif len(st.session_state.model.models) == 0:
                st.info("👈 Train the model first")
            else:
                if st.session_state.shap_explainer is None:
                    st.markdown("### 🔧 Initialize SHAP (one-time setup)")
                    if st.button("Initialize SHAP Explainer", type="primary"):
                        with st.spinner("Initializing..."):
                            try:
                                explainer = ClaimsShapExplainer(
                                    st.session_state.model,
                                    st.session_state.model.feature_names,
                                    st.session_state.config.data.categorical_features
                                )
                                explainer.create_explainers(st.session_state.data, max_samples=100)
                                st.session_state.shap_explainer = explainer
                                st.success("✅ Ready!")
                                st.balloons()
                            except Exception as e:
                                st.error(f"Error: {e}")
                else:
                    st.success("✅ SHAP explainer ready")
                    
                    mode = st.radio("Analysis Type:", ["Individual Claim", "Global Importance", "Top Frauds"], horizontal=True)
                    st.markdown("---")
                    
                    if mode == "Individual Claim":
                        st.subheader("🔍 Individual Claim - All Plot Types")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            idx = st.number_input("Claim Index", 0, len(st.session_state.data)-1,
                                                 int(st.session_state.fraud_scores.argmax()) if st.session_state.fraud_scores is not None else 0)
                        with col2:
                            if st.session_state.fraud_scores is not None:
                                st.metric("Score", f"{st.session_state.fraud_scores[idx]:,.0f}")
                        
                        target = st.selectbox("Target Feature", st.session_state.model.feature_names)
                        
                        # Multi-select for plot types
                        plot_types = st.multiselect(
                            "Select Plot Types to Generate",
                            ["🌊 Waterfall", "💪 Force", "📊 Bar", "🎯 Decision"],
                            default=["🌊 Waterfall", "💪 Force"],
                            help="Choose which SHAP visualizations to create"
                        )
                        
                        if st.button("🎯 Generate SHAP Explanations", type="primary"):
                            claim = st.session_state.data.iloc[[idx]]
                            with st.spinner("Computing SHAP values..."):
                                try:
                                    shap_vals, contrib = st.session_state.shap_explainer.explain_claim(claim, target, plot=False)
                                    
                                    # Get explainer info for advanced plots
                                    explainer_info = st.session_state.shap_explainer.explainers[target]
                                    predictor_features = explainer_info['predictor_features']
                                    shap_explainer = explainer_info['explainer']
                                    
                                    X_claim = claim[predictor_features].copy()
                                    X_claim = st.session_state.shap_explainer._preprocess_for_model(X_claim, predictor_features)
                                    
                                    expected_value = shap_explainer.expected_value
                                    if isinstance(expected_value, np.ndarray):
                                        expected_value = expected_value[0]
                                    
                                    if shap_vals.ndim > 1:
                                        sv = shap_vals[0]
                                    else:
                                        sv = shap_vals
                                    
                                    # ═══════════════════════════════════════
                                    # WATERFALL PLOT
                                    # ═══════════════════════════════════════
                                    if "🌊 Waterfall" in plot_types:
                                        st.markdown("### 🌊 Waterfall Plot")
                                        st.markdown("Shows step-by-step contribution from base to final prediction")
                                        
                                        top = contrib.head(15)
                                        colors = ['#d62728' if x > 0 else '#1f77b4' for x in top['shap_value']]
                                        
                                        fig = go.Figure(go.Bar(
                                            y=top['feature'], x=top['shap_value'], orientation='h',
                                            marker_color=colors, 
                                            text=[f"{v:+.4f}" for v in top['shap_value']], 
                                            textposition='outside',
                                            hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>'
                                        ))
                                        fig.update_layout(
                                            title=f"SHAP Waterfall: {target}",
                                            xaxis_title="SHAP Value (Impact on Prediction)",
                                            yaxis_title="Feature",
                                            height=600
                                        )
                                        fig.add_vline(x=0, line_color='black', line_width=1.5)
                                        fig.update_yaxes(autorange="reversed")
                                        st.plotly_chart(fig, use_container_width=True, key=f"waterfall_{idx}")
                                        
                                        st.info("🔴 Red = increases | 🔵 Blue = decreases")
                                    
                                    # ═══════════════════════════════════════
                                    # FORCE PLOT (Proper SHAP Native Visualization)
                                    # ═══════════════════════════════════════════
                                    if "💪 Force" in plot_types:
                                        st.markdown("### 💪 Force Plot")
                                        st.markdown("Interactive visualization showing how features push the prediction up or down")
                                        
                                        try:
                                            # First try native SHAP force plot
                                            import streamlit.components.v1 as components
                                            
                                            force_plot = shap.force_plot(
                                                expected_value,
                                                sv,
                                                X_claim.iloc[0],
                                                feature_names=predictor_features
                                            )
                                            
                                            # Display in Streamlit
                                            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                                            components.html(shap_html, height=400, scrolling=True)
                                            
                                            st.info(f"Base: {expected_value:.3f} → Prediction: {expected_value + sv.sum():.3f}")
                                            
                                        except Exception as native_error:
                                            st.info("Using interactive Plotly force plot")
                                            
                                            try:
                                                # Enhanced Plotly with proper shape handling
                                                top_n = min(15, len(sv))  # Safety: don't exceed array length
                                                top_indices = np.argsort(np.abs(sv))[-top_n:][::-1]
                                                
                                                # Safely get values with shape checking
                                                top_features = [predictor_features[i] for i in top_indices]
                                                
                                                # Get feature values safely
                                                X_values = X_claim.iloc[0].values
                                                if len(X_values) != len(predictor_features):
                                                    st.warning(f"Shape mismatch: {len(X_values)} values vs {len(predictor_features)} features")
                                                    # Use index-based access
                                                    top_values = [X_values[i] if i < len(X_values) else 0 for i in top_indices]
                                                else:
                                                    top_values = [X_values[i] for i in top_indices]
                                                
                                                top_shap = [sv[i] for i in top_indices]
                                                
                                                # Create figure
                                                fig_force = go.Figure()
                                                
                                                cumsum = expected_value
                                                annotations = []
                                                
                                                for i, (feat, val, shap_val) in enumerate(zip(top_features, top_values, top_shap)):
                                                    color = '#ff6b6b' if shap_val > 0 else '#4ecdc4'
                                                    
                                                    # Add bar segment
                                                    fig_force.add_trace(go.Bar(
                                                        x=[shap_val],
                                                        y=[0],
                                                        orientation='h',
                                                        base=cumsum,
                                                        marker=dict(
                                                            color=color,
                                                            line=dict(color='black', width=0.5)
                                                        ),
                                                        name=feat,
                                                        text=f"{feat[:12]}",
                                                        textposition='inside',
                                                        textfont=dict(size=8, color='white'),
                                                        hovertemplate=(
                                                            f"<b>{feat}</b><br>"
                                                            f"Value: {val:.3f}<br>"
                                                            f"SHAP: {shap_val:+.4f}<br>"
                                                            f"Impact: {cumsum:.3f} → {cumsum + shap_val:.3f}"
                                                            "<extra></extra>"
                                                        ),
                                                        showlegend=True
                                                    ))
                                                    
                                                    cumsum += shap_val
                                                
                                                # Add reference lines
                                                fig_force.add_vline(
                                                    x=expected_value,
                                                    line_dash="dash",
                                                    line_color="gray",
                                                    line_width=2,
                                                    annotation_text=f"Base: {expected_value:.2f}",
                                                    annotation_position="top"
                                                )
                                                
                                                fig_force.add_vline(
                                                    x=cumsum,
                                                    line_dash="dash",
                                                    line_color="green",
                                                    line_width=2,
                                                    annotation_text=f"Prediction: {cumsum:.2f}",
                                                    annotation_position="bottom"
                                                )
                                                
                                                # Layout
                                                fig_force.update_layout(
                                                    title=dict(
                                                        text=f"SHAP Force Plot: {target}<br><sub>Each bar shows a feature's contribution</sub>",
                                                        x=0.5,
                                                        xanchor='center',
                                                        font=dict(size=16)
                                                    ),
                                                    xaxis_title="Model Output Value",
                                                    yaxis_title="",
                                                    showlegend=True,
                                                    legend=dict(
                                                        orientation="v",
                                                        yanchor="top",
                                                        y=0.98,
                                                        xanchor="left",
                                                        x=1.02,
                                                        font=dict(size=9),
                                                        bgcolor="rgba(255,255,255,0.8)"
                                                    ),
                                                    height=400,
                                                    barmode='stack',
                                                    hovermode='closest',
                                                    plot_bgcolor='white',
                                                    paper_bgcolor='white'
                                                )
                                                
                                                fig_force.update_yaxes(showticklabels=False, range=[-0.5, 0.5])
                                                fig_force.update_xaxes(
                                                    gridcolor='lightgray',
                                                    gridwidth=0.5,
                                                    zeroline=True,
                                                    zerolinecolor='black',
                                                    zerolinewidth=1
                                                )
                                                
                                                st.plotly_chart(fig_force, use_container_width=True, key=f"force_{idx}_{target}")
                                                
                                                # Summary
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.metric("Base Value", f"{expected_value:.3f}")
                                                with col2:
                                                    st.metric("Prediction", f"{cumsum:.3f}")
                                                with col3:
                                                    change = cumsum - expected_value
                                                    st.metric("Change", f"{change:+.3f}", delta=f"{change:+.3f}")
                                                
                                                st.info("🔴 Red bars = push UP | 🔵 Blue bars = push DOWN | Hover over bars for details")
                                                
                                            except Exception as plot_error:
                                                st.error(f"Force plot error: {str(plot_error)}")
                                                import traceback
                                                with st.expander("Debug Info"):
                                                    st.code(traceback.format_exc())
                                                    st.write(f"sv shape: {sv.shape}")
                                                    st.write(f"X_claim shape: {X_claim.shape}")
                                                    st.write(f"predictor_features length: {len(predictor_features)}")
                                    
                                    if "📊 Bar" in plot_types:
                                        st.markdown("### 📊 Bar Plot (Feature Importance)")
                                        st.markdown("Ranked by absolute SHAP value (magnitude of impact)")
                                        
                                        abs_contrib = contrib.copy().sort_values('abs_shap', ascending=False).head(20)
                                        
                                        fig_bar = go.Figure(go.Bar(
                                            x=abs_contrib['abs_shap'],
                                            y=abs_contrib['feature'],
                                            orientation='h',
                                            marker_color=abs_contrib['abs_shap'],
                                            marker_colorscale='Reds',
                                            text=[f"{v:.4f}" for v in abs_contrib['abs_shap']],
                                            textposition='outside'
                                        ))
                                        
                                        fig_bar.update_layout(
                                            title=f"Feature Importance: {target}",
                                            xaxis_title="Mean Absolute SHAP",
                                            yaxis_title="Feature",
                                            height=600
                                        )
                                        fig_bar.update_yaxes(autorange="reversed")
                                        st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{idx}")
                                    
                                    # ═══════════════════════════════════════
                                    # DECISION PLOT
                                    # ═══════════════════════════════════════
                                    if "🎯 Decision" in plot_types:
                                        st.markdown("### 🎯 Decision Plot")
                                        st.markdown("Shows cumulative decision path from base to prediction")
                                        
                                        fig_dec, ax = plt.subplots(figsize=(10, 8))
                                        
                                        # Sort features by SHAP value
                                        sorted_idx = np.argsort(np.abs(sv))
                                        sorted_features = [predictor_features[i] for i in sorted_idx]
                                        sorted_shap = sv[sorted_idx]
                                        
                                        # Cumulative sum
                                        cumsum_vals = np.cumsum(sorted_shap)
                                        cumsum_vals = np.insert(cumsum_vals, 0, expected_value)
                                        
                                        y_pos = np.arange(len(sorted_features) + 1)
                                        ax.plot(cumsum_vals, y_pos, 'o-', linewidth=2, markersize=6, color='steelblue')
                                        
                                        ax.set_yticks(y_pos[1:])
                                        ax.set_yticklabels(sorted_features, fontsize=8)
                                        ax.set_xlabel('Model Output', fontsize=12, fontweight='bold')
                                        ax.set_title(f'Decision Plot: {target}', fontsize=14, fontweight='bold')
                                        ax.axvline(expected_value, color='gray', linestyle='--', alpha=0.5, label='Base')
                                        ax.grid(axis='x', alpha=0.3)
                                        ax.legend()
                                        
                                        st.pyplot(fig_dec, use_container_width=True)
                                        plt.close()
                                    
                                    # ═══════════════════════════════════════
                                    # DETAILED TABLE
                                    # ═══════════════════════════════════════
                                    st.markdown("---")
                                    st.markdown("### 📋 Detailed Contributions")
                                    st.dataframe(
                                        contrib[['feature','value','shap_value','abs_shap']].head(25),
                                        height=400
                                    )
                                    
                                except Exception as e:
                                    st.error(f"Error: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                    
                    elif mode == "Global Importance":
                        st.subheader("🌍 Global Feature Importance")
                        st.markdown("Computes mean absolute SHAP values across multiple claims to identify globally important features.")
                        
                        samples = st.slider("Number of samples to analyze", 100, 2000, 1000, 100,
                                          help="More samples = more accurate but slower")
                        
                        if st.button("🔍 Compute Global Importance", type="primary"):
                            with st.spinner(f"Computing SHAP values for {samples} samples... This may take 30-60 seconds"):
                                try:
                                    # Add progress feedback
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    status_text.text("Step 1/3: Preparing data...")
                                    progress_bar.progress(0.1)
                                    
                                    status_text.text(f"Step 2/3: Computing SHAP values for {samples} samples...")
                                    progress_bar.progress(0.3)
                                    
                                    # Call the method
                                    imp_df = st.session_state.shap_explainer.get_global_feature_importance(
                                        st.session_state.data, 
                                        max_samples=samples
                                    )
                                    
                                    status_text.text("Step 3/3: Creating visualization...")
                                    progress_bar.progress(0.8)
                                    
                                    # Display results
                                    top25 = imp_df.head(25)
                                    
                                    fig = go.Figure(go.Bar(
                                        x=top25['importance'], 
                                        y=top25['feature'], 
                                        orientation='h',
                                        marker_color=top25['importance'], 
                                        marker_colorscale='Reds',
                                        text=[f"{v:.4f}" for v in top25['importance']],
                                        textposition='outside'
                                    ))
                                    fig.update_layout(
                                        title="Top 25 Features (Mean |SHAP|)", 
                                        xaxis_title="Mean Absolute SHAP Value", 
                                        yaxis_title="Feature",
                                        height=700
                                    )
                                    fig.update_yaxes(autorange="reversed")
                                    
                                    progress_bar.progress(1.0)
                                    status_text.text("✅ Complete!")
                                    
                                    st.plotly_chart(fig, use_container_width=True, key="global_importance_chart")
                                    
                                    # Show full table
                                    st.markdown("### 📋 All Features Ranked")
                                    st.dataframe(imp_df, height=400)
                                    
                                    # Download button
                                    st.download_button(
                                        "📥 Download Full Results (CSV)", 
                                        imp_df.to_csv(index=False), 
                                        "shap_global_importance.csv", 
                                        "text/csv"
                                    )
                                    
                                    # Clear progress indicators
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                except Exception as e:
                                    st.error(f"❌ Error computing global importance: {str(e)}")
                                    st.error("**Debug Information:**")
                                    import traceback
                                    st.code(traceback.format_exc())
                                    
                                    # Show diagnostic info
                                    with st.expander("🔍 Diagnostic Information"):
                                        st.write("**Session State:**")
                                        st.write(f"- Model loaded: {st.session_state.model is not None}")
                                        st.write(f"- Data loaded: {st.session_state.data is not None}")
                                        if st.session_state.data is not None:
                                            st.write(f"- Data shape: {st.session_state.data.shape}")
                                        st.write(f"- SHAP explainer initialized: {st.session_state.shap_explainer is not None}")
                                        if st.session_state.shap_explainer is not None:
                                            st.write(f"- Number of explainers: {len(st.session_state.shap_explainer.explainers)}")
                                            st.write(f"- Feature names: {st.session_state.shap_explainer.feature_names}")
                    
                    else:  # Top Frauds
                        st.subheader("🚨 Top Frauds")
                        if st.session_state.fraud_scores is None:
                            st.warning("Compute fraud scores first")
                        else:
                            k = st.slider("Number of claims", 5, 50, 10)
                            if st.button("Generate", type="primary"):
                                with st.spinner(f"Processing top {k}..."):
                                    try:
                                        exp_df = st.session_state.shap_explainer.explain_top_frauds(
                                            st.session_state.data, st.session_state.fraud_scores, k)
                                        st.success(f"✅ Generated {k} explanations")
                                        st.dataframe(exp_df, height=500)
                                        st.download_button("📥 Download", exp_df.to_csv(index=False), f"top_{k}_shap.csv", "text/csv")
                                    except Exception as e:
                                        st.error(f"Error: {e}")
    
    # Tab: Model Monitoring (PSI)
    with tab_monitoring:
        st.header("📊 Model Monitoring - Data Drift Detection")
        st.markdown("""
        **Population Stability Index (PSI)** detects data drift between training and production data.
        
        **PSI Thresholds:**
        - PSI < 0.1: ✅ Stable (no action needed)
        - 0.1 ≤ PSI < 0.2: ⚠️ Minor drift (monitor closely)
        - PSI ≥ 0.2: 🚨 Major drift (consider retraining)
        """)
        
        if st.session_state.train_data is None or st.session_state.test_data is None:
            st.info("👈 Please load training and test data first from the sidebar")
            
            st.markdown("""
            ### Setup Instructions
            
            1. Load training data (this becomes your reference/baseline)
            2. The validation data loaded is used as "current" data for comparison
            3. Click "Analyze Data Drift" to compute PSI scores
            4. Review drift results and decide if retraining is needed
            """)
        else:
            # Configuration
            st.subheader("⚙️ Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Reference Data (Training)", f"{len(st.session_state.train_data):,} claims")
            
            with col2:
                st.metric("Current Data (Validation)", f"{len(st.session_state.data):,} claims")
            
            with col3:
                num_bins = st.selectbox("Number of Bins", [5, 10, 15, 20], index=1,
                                       help="More bins = finer granularity but needs more data")
            
            st.markdown("---")
            
            # Initialize PSI Monitor button
            if st.button("🔍 Analyze Data Drift", type="primary"):
                with st.spinner("Computing PSI scores..."):
                    try:
                        # Get numerical features only (PSI works best with numerical data)
                        cat_features = st.session_state.config.data.categorical_features
                        num_features = st.session_state.config.data.numerical_features
                        
                        # Prepare data - use only numerical features
                        train_numerical = st.session_state.train_data[num_features].values
                        current_numerical = st.session_state.data[num_features].values
                        
                        # Initialize PSI monitor
                        psi_monitor = PSIMonitor(
                            reference_data=train_numerical,
                            num_bins=num_bins,
                            feature_names=num_features
                        )
                        
                        # Detect drift
                        psi_results = psi_monitor.detect_drift(current_numerical)
                        
                        # Store in session state
                        st.session_state.psi_monitor = psi_monitor
                        st.session_state.psi_results = psi_results
                        
                        st.success("✅ PSI analysis complete!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Error computing PSI: {str(e)}")
                        import traceback
                        with st.expander("Debug Info"):
                            st.code(traceback.format_exc())
            
            # Display results if available
            if st.session_state.psi_results is not None:
                results = st.session_state.psi_results
                
                st.markdown("---")
                st.subheader("📈 Drift Detection Results")
                
                # Overall metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    overall_psi = results['overall_psi']
                    st.metric(
                        "Overall PSI",
                        f"{overall_psi:.4f}",
                        help="Average PSI across all features"
                    )
                
                with col2:
                    drift_status = results['drift_status']
                    status_emoji = {
                        'stable': '✅',
                        'minor': '⚠️',
                        'major': '🚨'
                    }
                    st.metric(
                        "Drift Status",
                        f"{status_emoji.get(drift_status, '❓')} {drift_status.upper()}",
                        help="Overall drift classification"
                    )
                
                with col3:
                    minor_drift = len(results['drifted_features']['minor'])
                    st.metric(
                        "Minor Drift",
                        f"{minor_drift}",
                        help="Features with 0.1 ≤ PSI < 0.2"
                    )
                
                with col4:
                    major_drift = len(results['drifted_features']['major'])
                    st.metric(
                        "Major Drift",
                        f"{major_drift}",
                        help="Features with PSI ≥ 0.2",
                        delta_color="inverse"
                    )
                
                # Recommendation
                st.markdown("---")
                st.subheader("💡 Recommendation")
                
                if drift_status == 'major':
                    st.error("""
                    **🚨 MAJOR DRIFT DETECTED**
                    
                    **Action Required:**
                    - **Retrain the model** with recent data
                    - Model performance may have significantly degraded
                    - Production predictions may be unreliable
                    
                    **Next Steps:**
                    1. Collect more recent training data
                    2. Retrain the model
                    3. Validate performance on held-out test set
                    4. Deploy updated model
                    """)
                elif drift_status == 'minor':
                    st.warning("""
                    **⚠️ MINOR DRIFT DETECTED**
                    
                    **Action Recommended:**
                    - **Monitor closely** over the next period
                    - Consider retraining if drift increases
                    - Check model performance metrics
                    
                    **Next Steps:**
                    1. Monitor fraud detection accuracy
                    2. Track PSI trends over time
                    3. Plan for retraining if drift worsens
                    """)
                else:
                    st.success("""
                    **✅ NO SIGNIFICANT DRIFT**
                    
                    **Current Status:**
                    - Model is stable
                    - No immediate action required
                    - Continue regular monitoring
                    
                    **Best Practices:**
                    - Run PSI analysis monthly/quarterly
                    - Track trends over time
                    - Maintain monitoring schedule
                    """)
                
                # PSI Scores by Feature
                st.markdown("---")
                st.subheader("📊 PSI Scores by Feature")
                
                psi_values = results['psi_values']
                psi_df = pd.DataFrame([
                    {'feature': feat, 'psi_score': psi}
                    for feat, psi in psi_values.items()
                ]).sort_values('psi_score', ascending=False)
                
                # Color code by drift level
                colors = []
                for psi in psi_df['psi_score']:
                    if psi >= 0.2:
                        colors.append('red')
                    elif psi >= 0.1:
                        colors.append('orange')
                    else:
                        colors.append('green')
                
                fig = go.Figure(go.Bar(
                    x=psi_df['psi_score'],
                    y=psi_df['feature'],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in psi_df['psi_score']],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>PSI: %{x:.4f}<extra></extra>'
                ))
                
                # Add threshold lines
                fig.add_vline(x=0.1, line_dash="dash", line_color="orange",
                             annotation_text="Minor (0.1)")
                fig.add_vline(x=0.2, line_dash="dash", line_color="red",
                             annotation_text="Major (0.2)")
                
                fig.update_layout(
                    title="PSI Scores by Feature",
                    xaxis_title="PSI Score",
                    yaxis_title="Feature",
                    height=max(400, len(psi_df) * 25),
                    showlegend=False
                )
                fig.update_yaxes(autorange="reversed")
                
                st.plotly_chart(fig, use_container_width=True, key="psi_scores_chart")
                
                # Legend
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("🟢 **Stable** (PSI < 0.1)")
                with col2:
                    st.markdown("🟠 **Minor Drift** (0.1 ≤ PSI < 0.2)")
                with col3:
                    st.markdown("🔴 **Major Drift** (PSI ≥ 0.2)")
                
                # Distribution Comparison
                st.markdown("---")
                st.subheader("📉 Distribution Comparison")
                st.markdown("Compare reference (training) vs current (validation) distributions for any feature")
                
                selected_feature = st.selectbox(
                    "Select Feature to Compare",
                    options=list(psi_values.keys()),
                    index=0
                )
                
                if selected_feature:
                    feature_idx = list(psi_values.keys()).index(selected_feature)
                    feature_psi = psi_values[selected_feature]
                    
                    # Get distributions
                    bin_edges, ref_proportions = st.session_state.psi_monitor.reference_distributions[feature_idx]
                    
                    train_numerical = st.session_state.train_data[list(psi_values.keys())].values
                    current_numerical = st.session_state.data[list(psi_values.keys())].values
                    
                    current_feature_data = current_numerical[:, feature_idx]
                    current_feature_data = current_feature_data[~np.isnan(current_feature_data)]
                    curr_counts, _ = np.histogram(current_feature_data, bins=bin_edges)
                    curr_proportions = curr_counts / curr_counts.sum()
                    
                    # Create comparison plot
                    x_labels = [f"Bin {i+1}" for i in range(len(ref_proportions))]
                    
                    fig_dist = go.Figure()
                    
                    fig_dist.add_trace(go.Bar(
                        x=x_labels,
                        y=ref_proportions,
                        name='Reference (Training)',
                        marker_color='steelblue',
                        opacity=0.7
                    ))
                    
                    fig_dist.add_trace(go.Bar(
                        x=x_labels,
                        y=curr_proportions,
                        name='Current (Validation)',
                        marker_color='coral',
                        opacity=0.7
                    ))
                    
                    fig_dist.update_layout(
                        title=f'Distribution Comparison: {selected_feature}<br><sub>PSI = {feature_psi:.4f}</sub>',
                        xaxis_title='Bin',
                        yaxis_title='Proportion',
                        barmode='group',
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True, key=f"dist_comparison_{selected_feature}")
                    
                    # Interpretation
                    if feature_psi >= 0.2:
                        st.error(f"🔴 **Major drift in {selected_feature}** - Distribution has changed significantly")
                    elif feature_psi >= 0.1:
                        st.warning(f"🟠 **Minor drift in {selected_feature}** - Distribution shows some changes")
                    else:
                        st.success(f"🟢 **Stable {selected_feature}** - Distribution is consistent")
                
                # Detailed PSI Table
                st.markdown("---")
                st.subheader("📋 Detailed PSI Scores")
                
                psi_df['status'] = psi_df['psi_score'].apply(
                    lambda x: 'Major Drift' if x >= 0.2 else ('Minor Drift' if x >= 0.1 else 'Stable')
                )
                
                st.dataframe(psi_df, height=400, use_container_width=True)
                
                # Download button
                st.download_button(
                    "📥 Download PSI Report (CSV)",
                    psi_df.to_csv(index=False),
                    "psi_drift_report.csv",
                    "text/csv",
                    help="Download full PSI analysis results"
                )
    
    # Tab: Fairness Analysis
    with tab_fairness:
        st.header("⚖️ Fairness Analysis - Bias Detection")
        st.markdown("""
        **Fairness Analysis** ensures the fraud detection model treats all groups equitably.
        
        **Key Metrics:**
        - **Disparate Impact Ratio**: Should be between 0.8 and 1.25 (close to 1.0 = fair)
        - **Flag Rate Parity**: Similar fraud flag rates across groups
        - **Statistical Significance**: p-value > 0.05 indicates no significant bias
        
        **Protected Attributes:** Gender, Age, Geographic Region, etc.
        """)
        
        if st.session_state.data is None or st.session_state.fraud_scores is None:
            st.info("👈 Please load data and compute fraud scores first")
            
            st.markdown("""
            ### Why Fairness Matters
            
            - **Legal Compliance**: Avoid discriminatory practices
            - **Ethical AI**: Ensure fair treatment of all individuals
            - **Reputation**: Build trust with stakeholders
            - **Effectiveness**: Unbiased models perform better
            
            ### How It Works
            
            1. Select protected attributes (e.g., gender, age)
            2. Set fraud detection threshold
            3. Analyze flag rates across groups
            4. Review disparate impact ratios
            5. Take action if bias detected
            """)
        else:
            # Configuration
            st.subheader("⚙️ Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Select protected attributes
                available_attributes = []
                
                # Detect potential protected attributes - PREFER BINNED AGE
                potential_attrs = [
                    'patient_gender', 'gender',
                    'geographic_region', 'region', 'state',
                    'claim_type', 'provider_specialty',
                    'race', 'ethnicity',
                    'income_level', 'socioeconomic_status'
                ]
                
                for attr in potential_attrs:
                    if attr in st.session_state.data.columns:
                        # Only include if it has reasonable number of groups (2-20)
                        n_unique = st.session_state.data[attr].nunique()
                        if 2 <= n_unique <= 20:
                            available_attributes.append(attr)
                
                # If patient_age exists, create age groups
                if 'patient_age' in st.session_state.data.columns:
                    st.session_state.data['patient_age_group'] = pd.cut(
                        st.session_state.data['patient_age'],
                        bins=[0, 30, 45, 60, 100],
                        labels=['<30', '30-45', '45-60', '60+']
                    )
                    available_attributes.insert(1, 'patient_age_group')  # Add after gender
                
                if not available_attributes:
                    st.warning("No protected attributes found. Using categorical features with 2-20 groups.")
                    for col in st.session_state.data.select_dtypes(include=['object', 'category']).columns:
                        n_unique = st.session_state.data[col].nunique()
                        if 2 <= n_unique <= 20:
                            available_attributes.append(col)
                            if len(available_attributes) >= 5:
                                break
                
                selected_attributes = st.multiselect(
                    "Select Protected Attributes",
                    options=available_attributes,
                    default=available_attributes[:min(3, len(available_attributes))],
                    help="Attributes to analyze for fairness (e.g., gender, age, region)"
                )
            
            with col2:
                threshold_pct = st.slider(
                    "Fraud Detection Threshold (Percentile)",
                    min_value=90.0,
                    max_value=99.9,
                    value=95.0,
                    step=0.5,
                    help="Claims above this percentile are flagged as potentially fraudulent"
                )
            
            st.markdown("---")
            
            # Run Fairness Analysis
            if st.button("⚖️ Run Fairness Analysis", type="primary"):
                if not selected_attributes:
                    st.error("Please select at least one protected attribute")
                else:
                    with st.spinner("Analyzing fairness across groups..."):
                        try:
                            from src.fairness_analysis import FairnessAnalyzer
                            
                            # Initialize analyzer
                            analyzer = FairnessAnalyzer(
                                data=st.session_state.data,
                                fraud_scores=st.session_state.fraud_scores,
                                protected_attributes=selected_attributes,
                                threshold_percentile=threshold_pct
                            )
                            
                            # Store in session state
                            st.session_state.fairness_analyzer = analyzer
                            st.session_state.fairness_results = analyzer.analyze_all_attributes()
                            
                            st.success("✅ Fairness analysis complete!")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"Error running fairness analysis: {str(e)}")
                            import traceback
                            with st.expander("Debug Info"):
                                st.code(traceback.format_exc())
            
            # Display results if available
            if hasattr(st.session_state, 'fairness_results') and st.session_state.fairness_results:
                results = st.session_state.fairness_results
                analyzer = st.session_state.fairness_analyzer
                
                st.markdown("---")
                st.subheader("📊 Fairness Overview")
                
                # Bias Summary Table
                bias_summary = analyzer.get_bias_summary()
                
                if not bias_summary.empty:
                    # Overall fairness status
                    all_fair = bias_summary['is_fair'].all()
                    
                    if all_fair:
                        st.success("✅ **No significant bias detected** across protected attributes")
                    else:
                        biased_attrs = bias_summary[~bias_summary['is_fair']]['attribute'].tolist()
                        st.error(f"⚠️ **Potential bias detected** in: {', '.join(biased_attrs)}")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Attributes Analyzed",
                            len(bias_summary),
                            help="Number of protected attributes checked"
                        )
                    
                    with col2:
                        fair_count = bias_summary['is_fair'].sum()
                        st.metric(
                            "Fair Attributes",
                            fair_count,
                            help="Attributes with no significant bias"
                        )
                    
                    with col3:
                        biased_count = (~bias_summary['is_fair']).sum()
                        st.metric(
                            "Biased Attributes",
                            biased_count,
                            help="Attributes with potential bias",
                            delta_color="inverse"
                        )
                    
                    with col4:
                        avg_fairness = bias_summary['fairness_score'].mean()
                        st.metric(
                            "Avg Fairness Score",
                            f"{avg_fairness:.3f}",
                            help="1.0 = perfectly fair, closer to 1.0 is better"
                        )
                    
                    # Detailed summary table
                    st.markdown("---")
                    st.subheader("📋 Fairness Summary by Attribute")
                    
                    display_summary = bias_summary.copy()
                    display_summary['fairness_status'] = display_summary['is_fair'].apply(
                        lambda x: '✅ Fair' if x else '⚠️ Biased'
                    )
                    display_summary = display_summary[[
                        'attribute', 'num_groups', 'min_di_ratio', 'max_di_ratio',
                        'fairness_score', 'fairness_status'
                    ]]
                    display_summary.columns = [
                        'Attribute', 'Groups', 'Min DI Ratio', 'Max DI Ratio',
                        'Fairness Score', 'Status'
                    ]
                    
                    st.dataframe(display_summary, use_container_width=True, height=300)
                    
                    st.info("""
                    **Disparate Impact (DI) Ratio Guide:**
                    - 0.8 to 1.25: ✅ Acceptable (fair)
                    - < 0.8 or > 1.25: ⚠️ Potential bias
                    - = 1.0: Perfect parity
                    """)
                
                # Detailed Analysis per Attribute
                st.markdown("---")
                st.subheader("🔍 Detailed Analysis")
                
                selected_attr = st.selectbox(
                    "Select Attribute for Detailed View",
                    options=list(results.keys()),
                    help="View detailed fairness metrics for specific attribute"
                )
                
                if selected_attr and selected_attr in results:
                    attr_results = results[selected_attr]
                    
                    if 'error' not in attr_results:
                        # Group comparison table
                        st.markdown(f"### 📊 Group Comparison: {selected_attr}")
                        
                        comparison_df = analyzer.get_detailed_comparison(selected_attr)
                        
                        if not comparison_df.empty:
                            # Format for display
                            display_df = comparison_df.copy()
                            display_df['flag_rate'] = display_df['flag_rate'].apply(lambda x: f"{x*100:.2f}%")
                            display_df['avg_score'] = display_df['avg_score'].apply(lambda x: f"{x:,.0f}")
                            display_df['median_score'] = display_df['median_score'].apply(lambda x: f"{x:,.0f}")
                            display_df['p95_score'] = display_df['p95_score'].apply(lambda x: f"{x:,.0f}")
                            display_df['p99_score'] = display_df['p99_score'].apply(lambda x: f"{x:,.0f}")
                            
                            display_df.columns = [
                                'Group', 'Count', 'Flagged', 'Flag Rate (%)',
                                'Avg Score', 'Median Score', 'P95 Score', 'P99 Score'
                            ]
                            
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Visualizations
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Flag rate comparison
                                fig_flag = go.Figure(go.Bar(
                                    x=comparison_df['group'],
                                    y=comparison_df['flag_rate'] * 100,
                                    marker_color='steelblue',
                                    text=[f"{v*100:.1f}%" for v in comparison_df['flag_rate']],
                                    textposition='outside'
                                ))
                                
                                fig_flag.update_layout(
                                    title=f"Flag Rate by {selected_attr}",
                                    xaxis_title="Group",
                                    yaxis_title="Flag Rate (%)",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_flag, use_container_width=True, key=f"flag_rate_{selected_attr}")
                            
                            with col2:
                                # Average score comparison
                                fig_score = go.Figure(go.Bar(
                                    x=comparison_df['group'],
                                    y=comparison_df['avg_score'],
                                    marker_color='coral',
                                    text=[f"{v:,.0f}" for v in comparison_df['avg_score']],
                                    textposition='outside'
                                ))
                                
                                fig_score.update_layout(
                                    title=f"Avg Fraud Score by {selected_attr}",
                                    xaxis_title="Group",
                                    yaxis_title="Average Score",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_score, use_container_width=True, key=f"avg_score_{selected_attr}")
                        
                        # Pairwise Comparisons
                        if 'pairwise_comparisons' in attr_results and attr_results['pairwise_comparisons']:
                            st.markdown("### 🔄 Pairwise Comparisons")
                            
                            comparisons = attr_results['pairwise_comparisons']
                            
                            for comp in comparisons:
                                with st.expander(f"{comp['group_a']} vs {comp['group_b']}", expanded=False):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric(
                                            f"{comp['group_a']} Flag Rate",
                                            f"{comp['rate_a']*100:.2f}%"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            f"{comp['group_b']} Flag Rate",
                                            f"{comp['rate_b']*100:.2f}%"
                                        )
                                    
                                    with col3:
                                        di_ratio = comp['disparate_impact_ratio']
                                        is_fair = comp['is_fair']
                                        
                                        st.metric(
                                            "DI Ratio",
                                            f"{di_ratio:.3f}",
                                            delta="✅ Fair" if is_fair else "⚠️ Biased"
                                        )
                                    
                                    # Statistical test results
                                    st.markdown("**Statistical Test:**")
                                    st.write(f"- Chi-square: {comp['chi_square']:.2f}")
                                    st.write(f"- p-value: {comp['p_value']:.4f}")
                                    st.write(f"- Effect size (Cohen's h): {comp['effect_size_h']:.3f}")
                                    
                                    if comp['p_value'] < 0.05:
                                        st.warning("⚠️ Statistically significant difference detected (p < 0.05)")
                                    else:
                                        st.success("✅ No statistically significant difference (p ≥ 0.05)")
                    
                    # Recommendations
                    st.markdown("---")
                    st.subheader("💡 Recommendations")
                    
                    if 'overall_metrics' in attr_results and attr_results['overall_metrics']:
                        if attr_results['overall_metrics']['is_fair']:
                            st.success(f"""
                            **✅ {selected_attr.upper()} APPEARS FAIR**
                            
                            - All disparate impact ratios within acceptable range (0.8-1.25)
                            - No significant statistical differences detected
                            - Continue regular monitoring
                            """)
                        else:
                            st.error(f"""
                            **⚠️ POTENTIAL BIAS IN {selected_attr.upper()}**
                            
                            **Immediate Actions:**
                            1. Review flagged cases from affected groups
                            2. Investigate root causes of disparity
                            3. Consider threshold adjustments
                            4. Retrain model with fairness constraints
                            
                            **Long-term Solutions:**
                            - Collect more diverse training data
                            - Use fairness-aware algorithms
                            - Implement bias mitigation techniques
                            - Regular fairness audits
                            """)
                
                # Export Report
                st.markdown("---")
                st.subheader("📄 Export Fairness Report")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download summary CSV
                    if not bias_summary.empty:
                        csv_summary = bias_summary.to_csv(index=False)
                        st.download_button(
                            "📥 Download Fairness Summary (CSV)",
                            csv_summary,
                            "fairness_summary.csv",
                            "text/csv",
                            help="Download summary of fairness metrics"
                        )
                
                with col2:
                    # Download detailed report
                    if selected_attr:
                        report_text = analyzer.generate_fairness_report(selected_attr)
                        st.download_button(
                            f"📥 Download {selected_attr} Report (TXT)",
                            report_text,
                            f"fairness_report_{selected_attr}.txt",
                            "text/plain",
                            help="Download detailed fairness analysis report"
                        )
    
    # Tab: Export
    with tab_export:
        st.header("📁 Export Results")
        
        if st.session_state.fraud_scores is not None:
            st.subheader("💾 Download Options")
            
            # Prepare export data
            export_df = st.session_state.data.copy()
            export_df['fraud_score'] = st.session_state.fraud_scores
            export_df['percentile'] = [
                (st.session_state.fraud_scores < score).sum() / len(st.session_state.fraud_scores) * 100
                for score in st.session_state.fraud_scores
            ]
            
            # Risk classification
            export_df['risk_level'] = pd.cut(
                export_df['percentile'],
                bins=[0, 90, 95, 99, 100],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Full export
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Claims (CSV)",
                    data=csv,
                    file_name="fraud_scores_all.csv",
                    mime="text/csv"
                )
            
            with col2:
                # High risk only
                high_risk = export_df[export_df['percentile'] >= 95]
                csv_high_risk = high_risk.to_csv(index=False)
                st.download_button(
                    label="📥 Download High Risk Only (CSV)",
                    data=csv_high_risk,
                    file_name="fraud_scores_high_risk.csv",
                    mime="text/csv"
                )
            
            # Summary statistics
            st.subheader("📊 Export Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Claims", len(export_df))
            
            with col2:
                st.metric("High Risk", len(export_df[export_df['risk_level'] == 'High']))
            
            with col3:
                st.metric("Critical", len(export_df[export_df['risk_level'] == 'Critical']))
            
            with col4:
                total_high_risk_amount = export_df[export_df['percentile'] >= 95]['claim_amount'].sum()
                st.metric("High Risk $", f"${total_high_risk_amount:,.0f}")
            
            # Preview
            st.subheader("👀 Export Preview")
            st.dataframe(export_df.head(20))
            
        else:
            st.info("Please compute fraud scores first")


if __name__ == "__main__":
    main()
