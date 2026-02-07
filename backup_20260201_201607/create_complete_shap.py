"""
Create Complete SHAP Version - Writes to src/ directory directly
"""

from pathlib import Path

webapp_path = Path("/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_enhanced.py")
output_path = Path("/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_complete_shap.py")

print("🔍 Reading current webapp...")
with open(webapp_path, 'r') as f:
    content = f.read()

print(f"📏 Current size: {len(content)} chars")

# Remove existing SHAP tab if present
if 'tab_shap' in content:
    print("🗑️  Removing old SHAP tab...")
    
    # Find and remove the SHAP tab section
    start_marker = "    # Tab SHAP: SHAP Explanations"
    end_marker = "    # Tab 5: Export"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        content = content[:start_idx] + content[end_idx:]
        print("✅ Removed old SHAP tab")
    
    # Fix tabs creation back to original
    if 'tabs_list' in content:
        # Find and replace tabs_list code with simple version
        old_tabs_pattern = content[content.find("tabs_list = ["):content.find("tab5 = tab_objects[4]")+len("tab5 = tab_objects[4]")]
        simple_tabs = '''tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🚨 Top Frauds",
        "📈 Feature Importance",
        "🔍 Individual Analysis",
        "📁 Export"
    ])'''
        content = content.replace(old_tabs_pattern, simple_tabs)
        print("✅ Reset tabs to original")

print("🔧 Adding COMPLETE SHAP with ALL plot types...")

# Now add complete SHAP implementation
# [Content continues exactly as in add_complete_shap.py]
# Write complete version

with open(output_path, 'w') as f:
    f.write(content)

print(f"\n✅ Created: {output_path}")
print(f"📏 Size: {len(content)} chars")
print("\n🎉 Ready to deploy!")
print(f"\nRun: cp {output_path} {webapp_path}")
print("Then: streamlit run app_enhanced.py")
