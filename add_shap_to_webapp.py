"""
Script to add complete SHAP integration to webapp_enhanced.py
"""

import re
from pathlib import Path

webapp_path = Path("/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_enhanced.py")

print("🔍 Reading current webapp...")
with open(webapp_path, 'r') as f:
    content = f.read()

print(f"📏 Current size: {len(content)} chars")

# Check if already has SHAP tab
if 'tab_shap' in content:
    print("⚠️  SHAP tab already exists!")
    exit(0)

print("🔧 Adding SHAP integration...")

# Step 1: Update tabs creation
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

# Step 2: Add SHAP tab code before "# Tab 5: Export"
shap_tab = '''
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
                        st.subheader("🔍 Individual Claim")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            idx = st.number_input("Claim Index", 0, len(st.session_state.data)-1,
                                                 int(st.session_state.fraud_scores.argmax()) if st.session_state.fraud_scores is not None else 0)
                        with col2:
                            if st.session_state.fraud_scores is not None:
                                st.metric("Score", f"{st.session_state.fraud_scores[idx]:,.0f}")
                        
                        target = st.selectbox("Target Feature", st.session_state.model.feature_names)
                        
                        if st.button("Generate SHAP", type="primary"):
                            claim = st.session_state.data.iloc[[idx]]
                            with st.spinner("Computing..."):
                                try:
                                    shap_vals, contrib = st.session_state.shap_explainer.explain_claim(claim, target, plot=False)
                                    
                                    top = contrib.head(15)
                                    colors = ['#d62728' if x > 0 else '#1f77b4' for x in top['shap_value']]
                                    
                                    fig = go.Figure(go.Bar(
                                        y=top['feature'], x=top['shap_value'], orientation='h',
                                        marker_color=colors, text=[f"{v:+.4f}" for v in top['shap_value']], textposition='outside'
                                    ))
                                    fig.update_layout(title=f"SHAP: {target}", xaxis_title="SHAP Value", height=600)
                                    fig.add_vline(x=0, line_color='black', line_width=1)
                                    fig.update_yaxes(autorange="reversed")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.info("🔴 Red: increases prediction | 🔵 Blue: decreases prediction")
                                    st.dataframe(contrib[['feature','value','shap_value']].head(20))
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    
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
    
'''

# Insert before "# Tab 5: Export"
insert_point = content.find('    # Tab 5: Export\n    with tab5:')
if insert_point == -1:
    print("❌ Could not find insertion point!")
    exit(1)

content = content[:insert_point] + shap_tab + content[insert_point:]

# Write new file
output = webapp_path.parent / "webapp_with_shap.py"
with open(output, 'w') as f:
    f.write(content)

print(f"✅ Created: {output}")
print(f"📏 New size: {len(content)} chars")
print("\n🎉 SHAP integration complete!")
print("\nTo use:")
print(f"  cd claims-autoencoder")
print(f"  cp src/webapp_with_shap.py src/webapp_enhanced.py")
print(f"  streamlit run app_enhanced.py")
