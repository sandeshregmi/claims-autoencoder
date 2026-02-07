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

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config_manager import ConfigManager
from src.data_ingestion import DataIngestion
from src.tree_models import ClaimsTreeAutoencoder
from src.psi_monitoring import PSIMonitor

# Try to import SHAP
try:
    import shap
    try:
        from shap_explainer import ClaimsShapExplainer
    except ImportError:
        from src.shap_explainer import ClaimsShapExplainer
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
                st.session_state.data = val_df
                st.success(f"✅ Loaded {len(val_df)} claims")
        
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
    
    # Main content tabs - conditionally add SHAP
    tabs_list = ["📊 Dashboard", "🚨 Top Frauds", "📈 Feature Importance", "🔍 Individual Analysis"]
    if SHAP_AVAILABLE:
        tabs_list.append("🔬 SHAP Explanations")
    tabs_list.append("📁 Export")
    
    tab_objects = st.tabs(tabs_list)
    tab1, tab2, tab3, tab4 = tab_objects[0], tab_objects[1], tab_objects[2], tab_objects[3]
    if SHAP_AVAILABLE:
        tab_shap, tab5 = tab_objects[4], tab_objects[5]
    else:
        tab5 = tab_objects[4]
    
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
                                    # FORCE PLOT
                                    # ═══════════════════════════════════════
                                    if "💪 Force" in plot_types:
                                        st.markdown("### 💪 Force Plot")
                                        st.markdown("Visualizes opposing forces pushing prediction higher/lower")
                                        
                                        import matplotlib.pyplot as plt
                                        fig_force, ax = plt.subplots(figsize=(14, 3))
                                        
                                        # Get top 10 features by absolute SHAP
                                        abs_shap = np.abs(sv)
                                        sorted_idx = np.argsort(abs_shap)[::-1][:10]
                                        
                                        base = expected_value
                                        cumsum = base
                                        
                                        for i in sorted_idx:
                                            shap_val = sv[i]
                                            color = '#d62728' if shap_val > 0 else '#1f77b4'
                                            ax.barh(0, shap_val, left=cumsum, height=0.6,
                                                   color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                                            cumsum += shap_val
                                        
                                        ax.axvline(base, color='gray', linestyle='--', linewidth=2, label=f'Base: {base:.3f}')
                                        ax.axvline(cumsum, color='green', linestyle='--', linewidth=2, label=f'Prediction: {cumsum:.3f}')
                                        
                                        ax.set_xlabel('Feature Impact', fontsize=12, fontweight='bold')
                                        ax.set_title(f'Force Plot: {target}', fontsize=14, fontweight='bold')
                                        ax.set_yticks([])
                                        ax.grid(axis='x', alpha=0.3)
                                        ax.legend()
                                        
                                        st.pyplot(fig_force, use_container_width=True)
                                        plt.close()
                                        
                                        st.info(f"Base: {base:.3f} → Prediction: {cumsum:.3f}")
                                    
                                    # ═══════════════════════════════════════
                                    # BAR PLOT
                                    # ═══════════════════════════════════════
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
                        st.subheader("🌍 Global Importance")
                        samples = st.slider("Samples", 100, 2000, 1000, 100)
                        
                        if st.button("Compute", type="primary"):
                            with st.spinner(f"Analyzing {samples} samples..."):
                                try:
                                    imp_df = st.session_state.shap_explainer.get_global_feature_importance(st.session_state.data, samples)
                                    top25 = imp_df.head(25)
                                    
                                    fig = go.Figure(go.Bar(x=top25['importance'], y=top25['feature'], orientation='h',
                                                          marker_color=top25['importance'], marker_colorscale='Reds'))
                                    fig.update_layout(title="Top 25 (Mean |SHAP|)", xaxis_title="Mean Absolute SHAP", height=700)
                                    fig.update_yaxes(autorange="reversed")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.dataframe(imp_df)
                                    st.download_button("📥 Download", imp_df.to_csv(index=False), "shap_importance.csv", "text/csv")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    
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
    
    # Tab 5: Export
    with tab5:
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
