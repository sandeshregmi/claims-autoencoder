"""
Streamlit Web Application for Claims Autoencoder
Interactive dashboard for model inference and monitoring.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
from typing import Optional
import json

from src.config_manager import ConfigManager
from src.preprocessing import ClaimsPreprocessor
from src.model_architecture import ClaimsAutoencoder
from src.evaluation import ModelEvaluator
from src.psi_monitoring import PSIMonitor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Claims Autoencoder Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Session state initialization
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'threshold' not in st.session_state:
    st.session_state.threshold = None
if 'reference_data' not in st.session_state:
    st.session_state.reference_data = None


def load_model_and_preprocessor(model_path: str, preprocessor_path: str, config_path: str):
    """Load model, preprocessor, and config."""
    try:
        # Load config
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        st.session_state.config = config

        # Load preprocessor
        preprocessor = ClaimsPreprocessor.load(preprocessor_path, config)
        st.session_state.preprocessor = preprocessor

        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.load(model_path, map_location=device)
        model.eval()
        st.session_state.model = model

        st.success("✅ Model and preprocessor loaded successfully!")
        return True

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return False


def score_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Score claims using loaded model."""
    if st.session_state.model is None or st.session_state.preprocessor is None:
        st.error("Model not loaded!")
        return None

    try:
        # Preprocess
        X = st.session_state.preprocessor.transform(df)

        # Score
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            reconstruction, encoding = st.session_state.model(X_tensor)
            errors = torch.mean((X_tensor - reconstruction) ** 2, dim=1)

        # Add results to dataframe
        result_df = df.copy()
        result_df['reconstruction_error'] = errors.numpy()

        if st.session_state.threshold is not None:
            result_df['is_anomaly'] = (errors.numpy() > st.session_state.threshold).astype(int)
            result_df['anomaly_score'] = errors.numpy() / st.session_state.threshold

        return result_df

    except Exception as e:
        st.error(f"Error scoring claims: {str(e)}")
        return None


def main():
    """Main application."""

    st.title("🏥 Claims Autoencoder Dashboard")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model loading
        st.subheader("Model Loading")

        model_path = st.text_input(
            "Model Path",
            value="models/best_model.pth"
        )

        preprocessor_path = st.text_input(
            "Preprocessor Path",
            value="models/preprocessor.pkl"
        )

        config_path = st.text_input(
            "Config Path",
            value="config/example_config.yaml"
        )

        if st.button("🔄 Load Model"):
            load_model_and_preprocessor(model_path, preprocessor_path, config_path)

        # Threshold setting
        st.subheader("Anomaly Detection")

        if st.session_state.config is not None:
            default_threshold = st.session_state.config.model.anomaly_threshold_percentile
        else:
            default_threshold = 95.0

        threshold_percentile = st.slider(
            "Threshold Percentile",
            min_value=90.0,
            max_value=99.9,
            value=default_threshold,
            step=0.1
        )

        # Upload reference data for threshold calculation
        st.subheader("Reference Data")
        ref_file = st.file_uploader(
            "Upload Training Data (for threshold)",
            type=['csv', 'parquet']
        )

        if ref_file is not None:
            try:
                if ref_file.name.endswith('.csv'):
                    ref_df = pd.read_csv(ref_file)
                else:
                    ref_df = pd.read_parquet(ref_file)

                st.session_state.reference_data = ref_df

                if st.session_state.model is not None:
                    # Calculate threshold
                    scored_ref = score_claims(ref_df)
                    if scored_ref is not None:
                        threshold = np.percentile(
                            scored_ref['reconstruction_error'],
                            threshold_percentile
                        )
                        st.session_state.threshold = threshold
                        st.success(f"✅ Threshold set: {threshold:.6f}")

            except Exception as e:
                st.error(f"Error loading reference data: {str(e)}")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Scoring",
        "📈 Monitoring",
        "🔍 Analysis",
        "ℹ️ Model Info"
    ])

    # Tab 1: Scoring
    with tab1:
        st.header("Claims Scoring")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Claims Data",
            type=['csv', 'parquet'],
            key='scoring_upload'
        )

        if uploaded_file is not None:
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_parquet(uploaded_file)

                st.write(f"Loaded {len(df)} claims")

                # Display sample
                with st.expander("View Data Sample"):
                    st.dataframe(df.head())

                # Score button
                if st.button("🎯 Score Claims"):
                    with st.spinner("Scoring claims..."):
                        scored_df = score_claims(df)

                    if scored_df is not None:
                        st.success("✅ Scoring completed!")

                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Total Claims",
                                len(scored_df)
                            )

                        with col2:
                            st.metric(
                                "Mean Error",
                                f"{scored_df['reconstruction_error'].mean():.4f}"
                            )

                        with col3:
                            if 'is_anomaly' in scored_df.columns:
                                anomaly_count = scored_df['is_anomaly'].sum()
                                st.metric("Anomalies", anomaly_count)

                        with col4:
                            if 'is_anomaly' in scored_df.columns:
                                anomaly_rate = scored_df['is_anomaly'].mean() * 100
                                st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

                        # Distribution plot
                        st.subheader("Reconstruction Error Distribution")
                        fig = px.histogram(
                            scored_df,
                            x='reconstruction_error',
                            nbins=50,
                            title="Distribution of Reconstruction Errors"
                        )

                        if st.session_state.threshold is not None:
                            fig.add_vline(
                                x=st.session_state.threshold,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Threshold"
                            )

                        st.plotly_chart(fig, use_container_width=True)

                        # Top anomalies
                        if 'is_anomaly' in scored_df.columns:
                            st.subheader("🚨 Top Anomalies")
                            top_anomalies = scored_df.nlargest(10, 'reconstruction_error')
                            st.dataframe(top_anomalies)

                        # Download results
                        st.subheader("💾 Download Results")
                        csv = scored_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="scored_claims.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Tab 2: Monitoring
    with tab2:
        st.header("Data Drift Monitoring")

        if st.session_state.reference_data is None:
            st.warning("⚠️ Please upload reference data in the sidebar first")
        else:
            current_file = st.file_uploader(
                "Upload Current Data",
                type=['csv', 'parquet'],
                key='monitoring_upload'
            )

            if current_file is not None:
                try:
                    # Load current data
                    if current_file.name.endswith('.csv'):
                        current_df = pd.read_csv(current_file)
                    else:
                        current_df = pd.read_parquet(current_file)

                    if st.button("🔍 Check for Drift"):
                        with st.spinner("Calculating PSI..."):
                            # Preprocess both datasets
                            ref_X = st.session_state.preprocessor.transform(
                                st.session_state.reference_data
                            )
                            curr_X = st.session_state.preprocessor.transform(current_df)

                            # Create PSI monitor
                            feature_names = st.session_state.preprocessor.get_feature_names()
                            monitor = PSIMonitor(ref_X, feature_names=feature_names)

                            # Calculate PSI
                            results = monitor.detect_drift(curr_X)

                            # Display results
                            st.subheader("Drift Detection Results")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Overall PSI", f"{results['overall_psi']:.4f}")

                            with col2:
                                drift_status = results['drift_status'].upper()
                                status_color = {
                                    'STABLE': '🟢',
                                    'MINOR': '🟡',
                                    'MAJOR': '🔴'
                                }
                                st.metric("Drift Status", f"{status_color.get(drift_status, '')} {drift_status}")

                            with col3:
                                st.metric(
                                    "Major Drift Features",
                                    len(results['drifted_features']['major'])
                                )

                            # PSI scores plot
                            st.subheader("PSI Scores by Feature")
                            psi_df = pd.DataFrame({
                                'Feature': list(results['psi_values'].keys()),
                                'PSI': list(results['psi_values'].values())
                            })

                            fig = px.bar(
                                psi_df,
                                x='Feature',
                                y='PSI',
                                title="Population Stability Index by Feature"
                            )
                            fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                                        annotation_text="Minor Drift")
                            fig.add_hline(y=0.2, line_dash="dash", line_color="red",
                                        annotation_text="Major Drift")

                            st.plotly_chart(fig, use_container_width=True)

                            # Drifted features
                            if results['drifted_features']['major']:
                                st.subheader("🔴 Features with Major Drift")
                                st.write(results['drifted_features']['major'])

                            if results['drifted_features']['minor']:
                                st.subheader("🟡 Features with Minor Drift")
                                st.write(results['drifted_features']['minor'])

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Tab 3: Analysis
    with tab3:
        st.header("Claims Analysis")

        analysis_file = st.file_uploader(
            "Upload Scored Claims",
            type=['csv', 'parquet'],
            key='analysis_upload'
        )

        if analysis_file is not None:
            try:
                if analysis_file.name.endswith('.csv'):
                    df = pd.read_csv(analysis_file)
                else:
                    df = pd.read_parquet(analysis_file)

                if 'reconstruction_error' in df.columns:
                    # Error distribution by category
                    categorical_cols = [col for col in df.columns
                                      if df[col].dtype == 'object' and col not in ['reconstruction_error', 'is_anomaly']]

                    if categorical_cols:
                        st.subheader("Error Distribution by Category")
                        selected_col = st.selectbox("Select Feature", categorical_cols)

                        fig = px.box(
                            df,
                            x=selected_col,
                            y='reconstruction_error',
                            title=f"Reconstruction Error by {selected_col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Correlation with error
                    numerical_cols = df.select_dtypes(include=[np.number]).columns
                    numerical_cols = [col for col in numerical_cols
                                    if col not in ['reconstruction_error', 'is_anomaly', 'anomaly_score']]

                    if len(numerical_cols) > 0:
                        st.subheader("Feature Correlations with Error")
                        correlations = df[numerical_cols + ['reconstruction_error']].corr()['reconstruction_error'].drop('reconstruction_error')
                        correlations = correlations.sort_values(ascending=False)

                        fig = px.bar(
                            x=correlations.values,
                            y=correlations.index,
                            orientation='h',
                            title="Correlation with Reconstruction Error"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Tab 4: Model Info
    with tab4:
        st.header("Model Information")

        if st.session_state.model is not None:
            st.subheader("Architecture")
            st.json({
                'input_dim': st.session_state.model.input_dim,
                'encoding_dim': st.session_state.model.encoding_dim,
                'hidden_layers': st.session_state.model.hidden_layers,
                'dropout_rate': st.session_state.model.dropout_rate,
                'total_parameters': st.session_state.model.count_parameters()
            })

            if st.session_state.config is not None:
                st.subheader("Configuration")
                st.json(st.session_state.config.dict())
        else:
            st.info("Load a model to view information")


if __name__ == "__main__":
    main()
