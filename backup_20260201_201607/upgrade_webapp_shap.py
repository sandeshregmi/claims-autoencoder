"""
Add Force Plot, Bar Plot, and Decision Plot to SHAP Tab in webapp_enhanced.py
This replaces the Individual Claim section with ALL plot types
"""

import sys
from pathlib import Path

webapp_path = Path("/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_enhanced.py")

print("🔧 Upgrading SHAP tab with ALL plot types...")
print()

with open(webapp_path, 'r') as f:
    content = f.read()

# Find the Individual Claim section
start_marker = 'if mode == "Individual Claim":'
end_marker = 'elif mode == "Global Importance":'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("❌ Could not find Individual Claim section!")
    sys.exit(1)

# New Individual Claim code with ALL plot types
new_individual_claim = '''if mode == "Individual Claim":
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
                    
                    el'''

# Replace the section
new_content = content[:start_idx] + new_individual_claim + content[end_idx:]

# Write to new file
output_path = Path("/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_enhanced_COMPLETE.py")
with open(output_path, 'w') as f:
    f.write(new_content)

print(f"✅ Created: {output_path}")
print()
print("📊 Added plot types:")
print("   ✅ Waterfall Plot")
print("   ✅ Force Plot (NEW!)")
print("   ✅ Bar Plot (NEW!)")
print("   ✅ Decision Plot (NEW!)")
print("   ✅ Multi-select checkboxes")
print()
print("🚀 To deploy:")
print(f"   cp {output_path} {webapp_path}")
print("   streamlit run app_enhanced.py")
