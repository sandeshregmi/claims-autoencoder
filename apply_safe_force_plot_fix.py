"""
Safe Force Plot Fix - Handles Array Shape Mismatches
"""

from pathlib import Path

webapp_path = Path("/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_enhanced.py")

print("🔧 Applying safe force plot fix...")

with open(webapp_path, 'r') as f:
    content = f.read()

# Find the force plot section
force_start = 'if "💪 Force" in plot_types:'
# Find next plot type as end marker
bar_start = 'if "📊 Bar" in plot_types:'

start_idx = content.find(force_start)
end_idx = content.find(bar_start)

if start_idx == -1 or end_idx == -1:
    print("❌ Could not find force plot section!")
    print(f"Start found: {start_idx != -1}, End found: {end_idx != -1}")
    exit(1)

# New safe force plot code with proper error handling
new_force_plot = '''if "💪 Force" in plot_types:
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
                                    
                                    '''

# Replace
new_content = content[:start_idx] + new_force_plot + content[end_idx:]

# Backup original
backup_path = webapp_path.parent / "webapp_enhanced_backup_before_force_fix.py"
with open(backup_path, 'w') as f:
    f.write(content)
print(f"📦 Backed up original to: {backup_path}")

# Write fixed version
with open(webapp_path, 'w') as f:
    f.write(new_content)

print("✅ Safe force plot applied!")
print("\n📊 New features:")
print("   • Shape mismatch protection")
print("   • Detailed error messages")
print("   • Interactive legend with feature names")
print("   • Hover tooltips with values")
print("   • Base and prediction lines")
print("   • Summary metrics below plot")
print("\n🔄 Restart Streamlit to see changes")
