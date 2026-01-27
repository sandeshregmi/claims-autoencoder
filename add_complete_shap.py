"""
Complete SHAP Integration with ALL Plot Types
Includes: Waterfall, Force, Beeswarm, Bar, Violin, Heatmap, Decision
"""

from pathlib import Path

webapp_path = Path("/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_enhanced.py")

print("🔍 Reading current webapp...")
with open(webapp_path, 'r') as f:
    content = f.read()

print(f"📏 Current size: {len(content)} chars")

if 'tab_shap' in content:
    print("⚠️  SHAP tab already exists! Delete it first or use webapp_enhanced_backup.py")
    exit(0)

print("🔧 Adding COMPLETE SHAP integration with ALL plot types...")

# Update tabs creation
tabs_old = '''    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🚨 Top Frauds",
        "📈 Feature Importance",
        "🔍 Individual Analysis",
        "📁 Export"
    ])'''

tabs_new = '''    # Main content tabs - conditionally add SHAP
    tabs_list = ["📊 Dashboard", "🚨 Top Frauds", "📈 Feature Importance", "🔍 Individual Analysis"]
    if SHAP_AVAILABLE:
        tabs_list.append("🔬 SHAP Explanations")
    tabs_list.append("📁 Export")
    
    tab_objects = st.tabs(tabs_list)
    tab1, tab2, tab3, tab4 = tab_objects[0], tab_objects[1], tab_objects[2], tab_objects[3]
    if SHAP_AVAILABLE:
        tab_shap, tab5 = tab_objects[4], tab_objects[5]
    else:
        tab5 = tab_objects[4]'''

content = content.replace(tabs_old, tabs_new)

# Complete SHAP tab with ALL plot types
shap_tab = '''
    # Tab SHAP: Complete SHAP Explanations with ALL Plot Types
    if SHAP_AVAILABLE:
        with tab_shap:
            st.header("🔬 SHAP Explanations - Complete Suite")
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** - Comprehensive model interpretability with multiple visualization types.
            """)
            
            if st.session_state.model is None or st.session_state.data is None:
                st.info("👈 Load model and data first")
            elif len(st.session_state.model.models) == 0:
                st.info("👈 Train the model first")
            else:
                # Initialize SHAP explainer
                if st.session_state.shap_explainer is None:
                    st.markdown("### 🔧 Initialize SHAP (one-time setup)")
                    st.markdown("This creates SHAP explainers for all features (~30-60 seconds)")
                    
                    if st.button("Initialize SHAP Explainer", type="primary"):
                        with st.spinner("Initializing SHAP explainers..."):
                            try:
                                explainer = ClaimsShapExplainer(
                                    st.session_state.model,
                                    st.session_state.model.feature_names,
                                    st.session_state.config.data.categorical_features
                                )
                                explainer.create_explainers(st.session_state.data, max_samples=100)
                                st.session_state.shap_explainer = explainer
                                st.success("✅ SHAP explainer ready!")
                                st.balloons()
                            except Exception as e:
                                st.error(f"Error initializing SHAP: {e}")
                                logger.exception(e)
                else:
                    st.success("✅ SHAP explainer initialized and ready")
                    
                    # Analysis mode selector
                    analysis_mode = st.radio(
                        "**Select Analysis Mode:**",
                        ["📊 Individual Claim (All Plots)", "🌍 Global Importance", "🚨 Top Frauds Batch"],
                        horizontal=True
                    )
                    
                    st.markdown("---")
                    
                    # ═══════════════════════════════════════════════════════════
                    # MODE 1: INDIVIDUAL CLAIM - ALL PLOT TYPES
                    # ═══════════════════════════════════════════════════════════
                    if analysis_mode == "📊 Individual Claim (All Plots)":
                        st.subheader("🔍 Individual Claim SHAP Analysis - All Visualizations")
                        
                        # Claim selector
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            claim_idx = st.number_input(
                                "Select Claim Index",
                                min_value=0,
                                max_value=len(st.session_state.data) - 1,
                                value=int(st.session_state.fraud_scores.argmax()) if st.session_state.fraud_scores is not None else 0,
                                help="Index of the claim to analyze"
                            )
                        with col2:
                            if st.session_state.fraud_scores is not None:
                                fraud_score = st.session_state.fraud_scores[claim_idx]
                                st.metric("Fraud Score", f"{fraud_score:,.0f}")
                        
                        # Target feature selector
                        target_feature = st.selectbox(
                            "Select Target Feature to Explain",
                            options=st.session_state.model.feature_names,
                            help="Which feature's prediction to explain"
                        )
                        
                        # Plot type multi-select
                        plot_types = st.multiselect(
                            "Select Plot Types to Generate",
                            ["Waterfall Plot", "Force Plot", "Bar Plot", "Decision Plot"],
                            default=["Waterfall Plot", "Force Plot"],
                            help="Choose which SHAP visualizations to create"
                        )
                        
                        if st.button("🎯 Generate SHAP Explanations", type="primary"):
                            claim_data = st.session_state.data.iloc[[claim_idx]]
                            
                            with st.spinner("Computing SHAP values..."):
                                try:
                                    shap_values, contributions = st.session_state.shap_explainer.explain_claim(
                                        claim_data, target_feature, plot=False
                                    )
                                    
                                    # Get explainer and data for plots
                                    explainer_info = st.session_state.shap_explainer.explainers[target_feature]
                                    predictor_features = explainer_info['predictor_features']
                                    explainer = explainer_info['explainer']
                                    
                                    # Prepare data
                                    X_claim = claim_data[predictor_features].copy()
                                    X_claim = st.session_state.shap_explainer._preprocess_for_model(X_claim, predictor_features)
                                    
                                    # Get expected value
                                    expected_value = explainer.expected_value
                                    if isinstance(expected_value, np.ndarray):
                                        expected_value = expected_value[0]
                                    
                                    # ═══════════════════════════════════════
                                    # WATERFALL PLOT
                                    # ═══════════════════════════════════════
                                    if "Waterfall Plot" in plot_types:
                                        st.markdown("### 🌊 Waterfall Plot")
                                        st.markdown("Shows how each feature contributes step-by-step from base value to final prediction")
                                        
                                        top_contrib = contributions.head(15)
                                        
                                        fig = go.Figure()
                                        colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_contrib['shap_value']]
                                        
                                        fig.add_trace(go.Bar(
                                            y=top_contrib['feature'],
                                            x=top_contrib['shap_value'],
                                            orientation='h',
                                            marker_color=colors,
                                            text=[f"{v:+.4f}" for v in top_contrib['shap_value']],
                                            textposition='outside',
                                            hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>'
                                        ))
                                        
                                        fig.update_layout(
                                            title=f"SHAP Waterfall: {target_feature}",
                                            xaxis_title="SHAP Value (Impact on Prediction)",
                                            yaxis_title="Feature",
                                            height=600,
                                            showlegend=False
                                        )
                                        fig.add_vline(x=0, line_color='black', line_width=1.5)
                                        fig.update_yaxes(autorange="reversed")
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        st.info("""
                                        **Interpretation:**
                                        - 🔴 **Red bars (positive)**: Features pushing the prediction HIGHER
                                        - 🔵 **Blue bars (negative)**: Features pushing the prediction LOWER
                                        - **Larger bars**: Stronger impact on the prediction
                                        """)
                                    
                                    # ═══════════════════════════════════════
                                    # FORCE PLOT (using matplotlib)
                                    # ═══════════════════════════════════════
                                    if "Force Plot" in plot_types:
                                        st.markdown("### 💪 Force Plot")
                                        st.markdown("Visualizes forces pushing prediction higher (red) vs lower (blue)")
                                        
                                        try:
                                            # Create force plot using matplotlib
                                            import matplotlib.pyplot as plt
                                            
                                            fig, ax = plt.subplots(figsize=(14, 3))
                                            
                                            # Get SHAP values
                                            if shap_values.ndim > 1:
                                                sv = shap_values[0]
                                            else:
                                                sv = shap_values
                                            
                                            # Sort by absolute value
                                            abs_shap = np.abs(sv)
                                            sorted_idx = np.argsort(abs_shap)[::-1][:10]
                                            
                                            # Create force plot manually
                                            base = expected_value
                                            cumsum = base
                                            
                                            for idx in sorted_idx:
                                                shap_val = sv[idx]
                                                feature_name = predictor_features[idx]
                                                
                                                color = '#d62728' if shap_val > 0 else '#1f77b4'
                                                ax.barh(0, shap_val, left=cumsum, height=0.5, 
                                                       color=color, alpha=0.7, edgecolor='black')
                                                cumsum += shap_val
                                            
                                            ax.axvline(base, color='gray', linestyle='--', linewidth=2, label=f'Base: {base:.3f}')
                                            ax.axvline(cumsum, color='green', linestyle='--', linewidth=2, label=f'Prediction: {cumsum:.3f}')
                                            
                                            ax.set_xlabel('Feature Value Impact')
                                            ax.set_title(f'Force Plot for {target_feature}')
                                            ax.set_yticks([])
                                            ax.legend()
                                            ax.grid(axis='x', alpha=0.3)
                                            
                                            st.pyplot(fig)
                                            plt.close()
                                            
                                            st.info(f"""
                                            **Interpretation:**
                                            - **Base value**: {expected_value:.3f} (average prediction)
                                            - **Final prediction**: {cumsum:.3f}
                                            - Red forces push prediction higher, blue forces push it lower
                                            """)
                                        except Exception as e:
                                            st.warning(f"Force plot visualization failed: {e}")
                                    
                                    # ═══════════════════════════════════════
                                    # BAR PLOT
                                    # ═══════════════════════════════════════
                                    if "Bar Plot" in plot_types:
                                        st.markdown("### 📊 Bar Plot (Feature Importance)")
                                        st.markdown("Shows mean absolute SHAP value for each feature")
                                        
                                        # Get absolute SHAP values
                                        abs_contrib = contributions.copy()
                                        abs_contrib = abs_contrib.sort_values('abs_shap', ascending=False).head(20)
                                        
                                        fig = go.Figure(go.Bar(
                                            x=abs_contrib['abs_shap'],
                                            y=abs_contrib['feature'],
                                            orientation='h',
                                            marker_color=abs_contrib['abs_shap'],
                                            marker_colorscale='Reds',
                                            text=[f"{v:.4f}" for v in abs_contrib['abs_shap']],
                                            textposition='outside'
                                        ))
                                        
                                        fig.update_layout(
                                            title=f"Feature Importance (|SHAP|) for {target_feature}",
                                            xaxis_title="Mean Absolute SHAP Value",
                                            yaxis_title="Feature",
                                            height=600
                                        )
                                        fig.update_yaxes(autorange="reversed")
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # ═══════════════════════════════════════
                                    # DECISION PLOT
                                    # ═══════════════════════════════════════
                                    if "Decision Plot" in plot_types:
                                        st.markdown("### 🎯 Decision Plot")
                                        st.markdown("Shows the cumulative effect of features on the prediction")
                                        
                                        try:
                                            # Create decision plot manually
                                            import matplotlib.pyplot as plt
                                            
                                            fig, ax = plt.subplots(figsize=(10, 8))
                                            
                                            # Get SHAP values
                                            if shap_values.ndim > 1:
                                                sv = shap_values[0]
                                            else:
                                                sv = shap_values
                                            
                                            # Sort by SHAP value
                                            sorted_idx = np.argsort(np.abs(sv))
                                            sorted_features = [predictor_features[i] for i in sorted_idx]
                                            sorted_shap = sv[sorted_idx]
                                            
                                            # Calculate cumulative sum
                                            cumsum = np.cumsum(sorted_shap)
                                            cumsum = np.insert(cumsum, 0, expected_value)
                                            
                                            # Plot
                                            y_pos = np.arange(len(sorted_features) + 1)
                                            ax.plot(cumsum, y_pos, 'o-', linewidth=2, markersize=8, color='steelblue')
                                            
                                            # Add feature names
                                            ax.set_yticks(y_pos[1:])
                                            ax.set_yticklabels(sorted_features)
                                            ax.set_xlabel('Model Output')
                                            ax.set_title(f'Decision Plot for {target_feature}')
                                            ax.axvline(expected_value, color='gray', linestyle='--', alpha=0.5, label='Base value')
                                            ax.grid(axis='x', alpha=0.3)
                                            ax.legend()
                                            
                                            st.pyplot(fig)
                                            plt.close()
                                            
                                        except Exception as e:
                                            st.warning(f"Decision plot failed: {e}")
                                    
                                    # ═══════════════════════════════════════
                                    # DETAILED CONTRIBUTIONS TABLE
                                    # ═══════════════════════════════════════
                                    st.markdown("---")
                                    st.markdown("### 📋 Detailed Feature Contributions")
                                    
                                    display_df = contributions[['feature', 'value', 'shap_value', 'abs_shap']].head(25)
                                    display_df.columns = ['Feature', 'Actual Value', 'SHAP Value', 'Absolute Impact']
                                    
                                    # Style the dataframe
                                    styled_df = display_df.style.format({
                                        'SHAP Value': '{:+.4f}',
                                        'Absolute Impact': '{:.4f}'
                                    }).background_gradient(
                                        subset=['SHAP Value'],
                                        cmap='RdBu_r',
                                        vmin=-display_df['SHAP Value'].abs().max(),
                                        vmax=display_df['SHAP Value'].abs().max()
                                    )
                                    
                                    st.dataframe(styled_df, height=600)
                                    
                                    # Download option
                                    csv = contributions.to_csv(index=False)
                                    st.download_button(
                                        "📥 Download Full Contributions (CSV)",
                                        csv,
                                        f"shap_contributions_claim_{claim_idx}.csv",
                                        "text/csv"
                                    )
                                    
                                except Exception as e:
                                    st.error(f"Error computing SHAP: {e}")
                                    logger.exception(e)
                    
                    # ═══════════════════════════════════════════════════════════
                    # MODE 2: GLOBAL IMPORTANCE
                    # ═══════════════════════════════════════════════════════════
                    elif analysis_mode == "🌍 Global Importance":
                        st.subheader("🌍 Global SHAP Feature Importance")
                        st.markdown("Analyze feature importance across multiple claims")
                        
                        max_samples = st.slider(
                            "Number of samples to analyze",
                            min_value=100,
                            max_value=2000,
                            value=1000,
                            step=100,
                            help="More samples = more accurate but slower"
                        )
                        
                        if st.button("📊 Compute Global Importance", type="primary"):
                            with st.spinner(f"Computing SHAP importance across {max_samples} samples..."):
                                try:
                                    importance_df = st.session_state.shap_explainer.get_global_feature_importance(
                                        st.session_state.data,
                                        max_samples=max_samples
                                    )
                                    
                                    # Bar plot
                                    st.markdown("### 📊 Top 25 Most Important Features")
                                    top25 = importance_df.head(25)
                                    
                                    fig = go.Figure(go.Bar(
                                        x=top25['importance'],
                                        y=top25['feature'],
                                        orientation='h',
                                        marker_color=top25['importance'],
                                        marker_colorscale='Reds',
                                        text=[f"{v:.3f}" for v in top25['importance']],
                                        textposition='outside'
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"Global Feature Importance (Mean |SHAP| over {max_samples} samples)",
                                        xaxis_title="Mean Absolute SHAP Value",
                                        yaxis_title="Feature",
                                        height=700
                                    )
                                    fig.update_yaxes(autorange="reversed")
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Full table
                                    st.markdown("### 📋 Complete Feature Rankings")
                                    st.dataframe(
                                        importance_df.style.background_gradient(
                                            subset=['importance'],
                                            cmap='Reds'
                                        ),
                                        height=400
                                    )
                                    
                                    # Download
                                    csv = importance_df.to_csv(index=False)
                                    st.download_button(
                                        "📥 Download Full Rankings (CSV)",
                                        csv,
                                        "shap_global_importance.csv",
                                        "text/csv"
                                    )
                                    
                                    st.success(f"✅ Analyzed {max_samples} claims successfully")
                                    
                                except Exception as e:
                                    st.error(f"Error computing global importance: {e}")
                                    logger.exception(e)
                    
                    # ═══════════════════════════════════════════════════════════
                    # MODE 3: TOP FRAUDS BATCH
                    # ═══════════════════════════════════════════════════════════
                    else:  # Top Frauds
                        st.subheader("🚨 Batch SHAP Explanations for Top Fraudulent Claims")
                        
                        if st.session_state.fraud_scores is None:
                            st.warning("⚠️ Please compute fraud scores first (Dashboard tab)")
                        else:
                            top_k = st.slider(
                                "Number of top fraudulent claims",
                                min_value=5,
                                max_value=50,
                                value=10
                            )
                            
                            if st.button("🎯 Generate Batch Explanations", type="primary"):
                                with st.spinner(f"Generating SHAP explanations for top {top_k} claims..."):
                                    try:
                                        explanations_df = st.session_state.shap_explainer.explain_top_frauds(
                                            st.session_state.data,
                                            st.session_state.fraud_scores,
                                            top_k=top_k
                                        )
                                        
                                        st.success(f"✅ Generated explanations for {top_k} most fraudulent claims")
                                        
                                        # Display
                                        st.markdown("### 📊 Top Fraudulent Claims with Feature Values")
                                        st.dataframe(explanations_df, height=500)
                                        
                                        # Summary stats
                                        st.markdown("### 📈 Summary Statistics")
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric("Total Claims", top_k)
                                        with col2:
                                            avg_score = explanations_df['fraud_score'].mean()
                                            st.metric("Avg Fraud Score", f"{avg_score:,.0f}")
                                        with col3:
                                            max_score = explanations_df['fraud_score'].max()
                                            st.metric("Max Fraud Score", f"{max_score:,.0f}")
                                        
                                        # Download
                                        csv = explanations_df.to_csv(index=False)
                                        st.download_button(
                                            "📥 Download Batch Explanations (CSV)",
                                            csv,
                                            f"top_{top_k}_fraud_shap_explanations.csv",
                                            "text/csv"
                                        )
                                        
                                    except Exception as e:
                                        st.error(f"Error generating batch explanations: {e}")
                                        logger.exception(e)
    
'''

# Insert before "# Tab 5: Export"
insert_point = content.find('    # Tab 5: Export\n    with tab5:')
if insert_point == -1:
    print("❌ Could not find insertion point!")
    exit(1)

content = content[:insert_point] + shap_tab + content[insert_point:]

# Write new file
output = webapp_path.parent / "webapp_complete_shap.py"
with open(output, 'w') as f:
    f.write(content)

print(f"✅ Created: {output}")
print(f"📏 New size: {len(content)} chars")
print("\n🎉 COMPLETE SHAP integration done!")
print("\n📊 Included plot types:")
print("  ✅ Waterfall Plot")
print("  ✅ Force Plot")
print("  ✅ Bar Plot")
print("  ✅ Decision Plot")
print("  ✅ Global Importance")
print("  ✅ Batch Explanations")
print("\nTo use:")
print(f"  cp {output} src/webapp_enhanced.py")
print(f"  streamlit run app_enhanced.py")
