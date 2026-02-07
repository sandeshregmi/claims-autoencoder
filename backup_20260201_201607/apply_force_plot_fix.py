"""
Automated Fix: Replace Force Plot with Proper SHAP Visualization
"""

from pathlib import Path
import re

webapp_path = Path("/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_enhanced.py")

print("🔧 Fixing Force Plot to show proper SHAP visualization...")

with open(webapp_path, 'r') as f:
    content = f.read()

# Find the force plot section
force_plot_start = '# FORCE PLOT'
force_plot_end = '# BAR PLOT'

start_idx = content.find(force_plot_start)
end_idx = content.find(force_plot_end)

if start_idx == -1 or end_idx == -1:
    print("❌ Could not find force plot section!")
    exit(1)

# New proper force plot code
new_force_plot = '''# FORCE PLOT (Proper SHAP Native Visualization)
                                    # ═══════════════════════════════════════════
                                    if "💪 Force" in plot_types:
                                        st.markdown("### 💪 Force Plot")
                                        st.markdown("Interactive SHAP force plot with individual feature labels")
                                        
                                        try:
                                            # Try native SHAP force plot first
                                            import streamlit.components.v1 as components
                                            
                                            force_plot = shap.force_plot(
                                                expected_value,
                                                sv,
                                                X_claim.iloc[0],
                                                feature_names=predictor_features
                                            )
                                            
                                            # Render in Streamlit
                                            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                                            components.html(shap_html, height=400, scrolling=True)
                                            
                                            st.info(f"Base: {expected_value:.3f} → Prediction: {expected_value + sv.sum():.3f}")
                                            
                                        except Exception as e:
                                            st.warning(f"Native SHAP force plot failed, using enhanced fallback")
                                            
                                            # Enhanced Plotly force plot with feature labels
                                            # Get top 15 features by absolute SHAP
                                            top_indices = np.argsort(np.abs(sv))[-15:][::-1]
                                            top_features = [predictor_features[i] for i in top_indices]
                                            top_values = X_claim.iloc[0].values[top_indices]
                                            top_shap = sv[top_indices]
                                            
                                            fig_force = go.Figure()
                                            
                                            cumsum = expected_value
                                            for i, (feat, val, shap_val) in enumerate(zip(top_features, top_values, top_shap)):
                                                color = '#ff6b6b' if shap_val > 0 else '#4ecdc4'
                                                
                                                fig_force.add_trace(go.Bar(
                                                    x=[shap_val],
                                                    y=[0],
                                                    orientation='h',
                                                    base=cumsum,
                                                    marker_color=color,
                                                    marker_line_color='black',
                                                    marker_line_width=0.5,
                                                    name=f"{feat}",
                                                    text=f"{feat[:15]}<br>{shap_val:+.2f}",
                                                    textposition='inside',
                                                    textfont=dict(size=9, color='white'),
                                                    hovertemplate=(
                                                        f"<b>{feat}</b><br>"
                                                        f"Value: {val:.3f}<br>"
                                                        f"SHAP: {shap_val:+.4f}<br>"
                                                        f"Position: {cumsum:.3f} → {cumsum + shap_val:.3f}"
                                                        "<extra></extra>"
                                                    )
                                                ))
                                                cumsum += shap_val
                                            
                                            # Add base and prediction lines
                                            fig_force.add_vline(
                                                x=expected_value, 
                                                line_dash="dash", 
                                                line_color="gray",
                                                annotation_text=f"Base: {expected_value:.2f}",
                                                annotation_position="top"
                                            )
                                            fig_force.add_vline(
                                                x=cumsum, 
                                                line_dash="dash", 
                                                line_color="green",
                                                annotation_text=f"Pred: {cumsum:.2f}",
                                                annotation_position="top"
                                            )
                                            
                                            fig_force.update_layout(
                                                title=dict(
                                                    text=f"SHAP Force Plot: {target}<br><sub>Red = increases | Blue = decreases</sub>",
                                                    x=0.5,
                                                    xanchor='center'
                                                ),
                                                xaxis_title="Model Output Value",
                                                yaxis_title="",
                                                showlegend=True,
                                                legend=dict(
                                                    orientation="v",
                                                    yanchor="top",
                                                    y=1,
                                                    xanchor="left",
                                                    x=1.02,
                                                    font=dict(size=9)
                                                ),
                                                height=500,
                                                barmode='stack',
                                                hovermode='closest'
                                            )
                                            fig_force.update_yaxes(showticklabels=False, range=[-0.5, 0.5])
                                            fig_force.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
                                            
                                            st.plotly_chart(fig_force, use_container_width=True, key=f"force_{idx}")
                                            
                                            st.info(f"📊 Shows top 15 features by absolute SHAP value | Base: {expected_value:.3f} → Prediction: {cumsum:.3f}")
                                    
                                    # '''

# Replace the section
new_content = content[:start_idx] + new_force_plot + content[end_idx:]

# Write back
with open(webapp_path, 'w') as f:
    f.write(new_content)

print("✅ Force plot fixed!")
print("\n📊 New force plot features:")
print("   • Individual feature labels")
print("   • Feature values in hover")
print("   • Color-coded bars (red=increase, blue=decrease)")
print("   • Interactive legend")
print("   • Base and prediction lines")
print("\n🔄 Restart your Streamlit app to see changes:")
print("   streamlit run app_enhanced.py")
