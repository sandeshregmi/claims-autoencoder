# PSI MONITORING TAB CODE
# Insert this code after the SHAP tab and before the Export tab

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
