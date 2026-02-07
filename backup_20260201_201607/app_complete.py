"""
Enhanced Streamlit Web Application for Claims Fraud Detection
Integrated with Tree Models, PSI Monitoring, and Fairness Analysis
"""

import streamlit as st
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Add project root to path
sys.path.insert(0, '/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder')

from src.config_manager import ConfigManager
from src.data_ingestion import DataIngestion
from src.tree_models import ClaimsTreeAutoencoder
from src.psi_monitoring import PSIMonitor
from src.fairness_analysis import FairnessAnalyzer

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
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'psi_results' not in st.session_state:
    st.session_state.psi_results = None
if 'fairness_results' not in st.session_state:
    st.session_state.fairness_results = None

def main():
    st.title("🚨 Claims Fraud Detection Dashboard")
    st.markdown("**Complete system with PSI Monitoring and Fairness Analysis**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_type = st.selectbox("Select Model", options=['catboost', 'xgboost'], index=0)
        
        if st.button("🔄 Load Model & Data", type="primary"):
            with st.spinner("Loading..."):
                try:
                    config_manager = ConfigManager('config/example_config.yaml')
                    config = config_manager.get_config()
                    
                    model = ClaimsTreeAutoencoder(model_type=model_type)
                    
                    data_ingestion = DataIngestion(config)
                    train_df, val_df, test_df = data_ingestion.load_train_val_test()
                    
                    st.session_state.model = model
                    st.session_state.config = config
                    st.session_state.train_data = train_df
                    st.session_state.data = val_df
                    st.session_state.test_data = test_df
                    
                    st.success("✅ Model and data loaded!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if st.session_state.model and st.session_state.data is not None:
            if st.button("🎓 Train Model"):
                with st.spinner("Training..."):
                    cat_features = st.session_state.config.data.categorical_features
                    num_features = st.session_state.config.data.numerical_features
                    st.session_state.model.fit(st.session_state.data, cat_features, num_features, verbose=False)
                    st.success("✅ Model trained!")
            
            if len(st.session_state.model.models) > 0:
                if st.button("🎯 Compute Fraud Scores"):
                    with st.spinner("Computing scores..."):
                        fraud_scores, per_feature_errors = st.session_state.model.compute_fraud_scores(st.session_state.data)
                        st.session_state.fraud_scores = fraud_scores
                        st.session_state.per_feature_errors = per_feature_errors
                        st.success("✅ Scores computed!")
    
    # Main tabs
    tabs = st.tabs(["📊 Dashboard", "📊 PSI Monitoring", "⚖️ Fairness Analysis", "📁 Export"])
    
    # Dashboard tab
    with tabs[0]:
        st.header("Fraud Detection Overview")
        
        if st.session_state.fraud_scores is not None:
            fraud_scores = st.session_state.fraud_scores
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Claims", f"{len(fraud_scores):,}")
            with col2:
                st.metric("Mean Score", f"{fraud_scores.mean():,.0f}")
            with col3:
                p95 = np.percentile(fraud_scores, 95)
                flagged_95 = (fraud_scores > p95).sum()
                st.metric("High Risk (95th %)", f"{flagged_95:,}")
            with col4:
                p99 = np.percentile(fraud_scores, 99)
                flagged_99 = (fraud_scores > p99).sum()
                st.metric("Critical (99th %)", f"{flagged_99:,}")
            
            # Distribution
            fig = go.Figure(go.Histogram(x=fraud_scores, nbinsx=50))
            fig.update_layout(title="Fraud Score Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Please load model, train, and compute scores from sidebar")
    
    # PSI Monitoring tab
    with tabs[1]:
        st.header("📊 Model Monitoring - Data Drift Detection")
        
        if st.session_state.train_data is not None and st.session_state.data is not None:
            num_bins = st.selectbox("Number of Bins", [5, 10, 15, 20], index=1)
            
            if st.button("🔍 Analyze Data Drift", type="primary"):
                with st.spinner("Computing PSI..."):
                    try:
                        num_features = st.session_state.config.data.numerical_features
                        train_numerical = st.session_state.train_data[num_features].values
                        current_numerical = st.session_state.data[num_features].values
                        
                        psi_monitor = PSIMonitor(
                            reference_data=train_numerical,
                            num_bins=num_bins,
                            feature_names=num_features
                        )
                        
                        psi_results = psi_monitor.detect_drift(current_numerical)
                        st.session_state.psi_results = psi_results
                        
                        st.success("✅ PSI analysis complete!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            if st.session_state.psi_results:
                results = st.session_state.psi_results
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall PSI", f"{results['overall_psi']:.4f}")
                with col2:
                    st.metric("Drift Status", results['drift_status'].upper())
                with col3:
                    major_drift = len(results['drifted_features']['major'])
                    st.metric("Major Drift Features", f"{major_drift}")
                
                # PSI scores chart
                psi_values = results['psi_values']
                psi_df = pd.DataFrame([
                    {'feature': feat, 'psi_score': psi}
                    for feat, psi in psi_values.items()
                ]).sort_values('psi_score', ascending=False)
                
                colors = ['red' if psi >= 0.2 else 'orange' if psi >= 0.1 else 'green' for psi in psi_df['psi_score']]
                
                fig = go.Figure(go.Bar(
                    x=psi_df['psi_score'],
                    y=psi_df['feature'],
                    orientation='h',
                    marker_color=colors
                ))
                fig.update_layout(title="PSI Scores by Feature", height=600)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Please load training and validation data first")
    
    # Fairness Analysis tab
    with tabs[2]:
        st.header("⚖️ Fairness Analysis - Bias Detection")
        
        if st.session_state.data is not None and st.session_state.fraud_scores is not None:
            # Detect available attributes
            available_attributes = []
            potential_attrs = ['patient_gender', 'gender', 'geographic_region', 'region', 'claim_type']
            
            for attr in potential_attrs:
                if attr in st.session_state.data.columns:
                    n_unique = st.session_state.data[attr].nunique()
                    if 2 <= n_unique <= 20:
                        available_attributes.append(attr)
            
            # Add age groups
            if 'patient_age' in st.session_state.data.columns:
                st.session_state.data['patient_age_group'] = pd.cut(
                    st.session_state.data['patient_age'],
                    bins=[0, 30, 45, 60, 100],
                    labels=['<30', '30-45', '45-60', '60+']
                )
                available_attributes.insert(0, 'patient_age_group')
            
            if available_attributes:
                col1, col2 = st.columns(2)
                with col1:
                    selected_attributes = st.multiselect(
                        "Select Protected Attributes",
                        options=available_attributes,
                        default=available_attributes[:min(2, len(available_attributes))]
                    )
                with col2:
                    threshold_pct = st.slider("Fraud Detection Threshold (%ile)", 90.0, 99.9, 95.0, 0.5)
                
                if selected_attributes and st.button("⚖️ Run Fairness Analysis", type="primary"):
                    with st.spinner("Analyzing fairness..."):
                        try:
                            analyzer = FairnessAnalyzer(
                                data=st.session_state.data,
                                fraud_scores=st.session_state.fraud_scores,
                                protected_attributes=selected_attributes,
                                threshold_percentile=threshold_pct
                            )
                            
                            st.session_state.fairness_results = analyzer.analyze_all_attributes()
                            st.success("✅ Fairness analysis complete!")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                if st.session_state.fairness_results:
                    results = st.session_state.fairness_results
                    
                    for attr, attr_results in results.items():
                        if 'error' not in attr_results:
                            st.subheader(f"📊 {attr}")
                            
                            if 'group_metrics' in attr_results:
                                group_df = pd.DataFrame(attr_results['group_metrics'])
                                st.dataframe(group_df)
                            
                            if 'pairwise_comparisons' in attr_results:
                                for comp in attr_results['pairwise_comparisons'][:3]:  # Show top 3
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(f"{comp['group_a']}", f"{comp['rate_a']*100:.1f}%")
                                    with col2:
                                        st.metric(f"{comp['group_b']}", f"{comp['rate_b']*100:.1f}%")
                                    with col3:
                                        is_fair = comp['is_fair']
                                        st.metric("DI Ratio", f"{comp['disparate_impact_ratio']:.3f}",
                                                delta="✅ Fair" if is_fair else "⚠️ Biased")
            else:
                st.warning("No protected attributes found in data")
        else:
            st.info("👈 Please load data and compute fraud scores first")
    
    # Export tab
    with tabs[3]:
        st.header("📁 Export Results")
        
        if st.session_state.fraud_scores is not None:
            export_df = st.session_state.data.copy()
            export_df['fraud_score'] = st.session_state.fraud_scores
            export_df['percentile'] = [
                (st.session_state.fraud_scores < score).sum() / len(st.session_state.fraud_scores) * 100
                for score in st.session_state.fraud_scores
            ]
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                "📥 Download All Claims (CSV)",
                csv,
                "fraud_scores_all.csv",
                "text/csv"
            )
            
            st.dataframe(export_df.head(20))
        else:
            st.info("No data to export yet")

if __name__ == "__main__":
    main()
