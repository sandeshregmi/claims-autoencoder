"""
Fix Force Plot to Use Native SHAP Visualization
This creates the proper interactive SHAP force plot with feature labels
"""

# Instructions for fixing the force plot in webapp_enhanced.py
print("""
╔════════════════════════════════════════════════════════════════════╗
║  FIX: Replace Custom Force Plot with Native SHAP Force Plot       ║
╚════════════════════════════════════════════════════════════════════╝

The current force plot is a simple bar chart. 
A proper SHAP force plot shows:
  • Individual feature names as labels
  • Feature values
  • Interactive HTML visualization
  • Proper SHAP force plot layout

SOLUTION: Use SHAP's native force_plot() function

Find this section in webapp_enhanced.py (around line 820-850):
──────────────────────────────────────────────────────────────────

                    # FORCE PLOT
                    if "💪 Force" in plot_types:
                        st.markdown("### 💪 Force Plot")
                        ...
                        # Current code with matplotlib bars
                        
──────────────────────────────────────────────────────────────────

REPLACE with:
──────────────────────────────────────────────────────────────────

                    # ═══════════════════════════════════════════
                    # FORCE PLOT (Native SHAP)
                    # ═══════════════════════════════════════════
                    if "💪 Force" in plot_types:
                        st.markdown("### 💪 Force Plot")
                        st.markdown("Interactive SHAP force plot showing individual feature contributions")
                        
                        try:
                            # Use SHAP's native force plot
                            import shap
                            
                            # Generate HTML force plot
                            force_plot = shap.force_plot(
                                expected_value,
                                sv,
                                X_claim.iloc[0],
                                feature_names=predictor_features
                            )
                            
                            # Save to HTML and display in Streamlit
                            import streamlit.components.v1 as components
                            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                            components.html(shap_html, height=400)
                            
                            st.info(f"Base value: {expected_value:.3f} → Prediction: {expected_value + sv.sum():.3f}")
                            
                        except Exception as e:
                            st.error(f"Could not generate native force plot: {e}")
                            st.markdown("**Fallback: Showing simplified visualization**")
                            
                            # Fallback to interactive Plotly
                            import plotly.graph_objects as go
                            
                            # Get top features
                            top_indices = np.argsort(np.abs(sv))[-15:][::-1]
                            top_features = [predictor_features[i] for i in top_indices]
                            top_values = X_claim.iloc[0].values[top_indices]
                            top_shap = sv[top_indices]
                            
                            # Create stacked bar with labels
                            fig = go.Figure()
                            
                            cumsum = expected_value
                            for feat, val, shap_val in zip(top_features, top_values, top_shap):
                                color = 'red' if shap_val > 0 else 'blue'
                                fig.add_trace(go.Bar(
                                    x=[shap_val],
                                    y=[0],
                                    orientation='h',
                                    base=cumsum,
                                    marker_color=color,
                                    name=f"{feat} = {val:.2f}",
                                    text=f"{feat}<br>{shap_val:+.3f}",
                                    textposition='inside',
                                    hovertemplate=f"<b>{feat}</b><br>Value: {val:.2f}<br>SHAP: {shap_val:+.4f}<extra></extra>"
                                ))
                                cumsum += shap_val
                            
                            fig.update_layout(
                                title=f"Force Plot: {target}",
                                xaxis_title="Model Output",
                                yaxis_title="",
                                showlegend=True,
                                height=500,
                                barmode='stack'
                            )
                            fig.update_yaxes(showticklabels=False)
                            
                            st.plotly_chart(fig, use_container_width=True, key=f"force_{idx}")

──────────────────────────────────────────────────────────────────
""")

print("\n✅ Instructions shown above")
print("\nOR - Run this automated fix:")
print("  python apply_force_plot_fix.py")
