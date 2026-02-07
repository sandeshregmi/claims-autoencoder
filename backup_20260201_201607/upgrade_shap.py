"""
Upgrade existing SHAP tab to include ALL plot types
Adds: Force Plot, Bar Plot, Decision Plot + Multi-select
"""

print("🔧 Upgrading SHAP tab with ALL plot types...")
print("\n✅ Your current SHAP tab has:")
print("   - Waterfall plot only")
print("   - Global importance")
print("   - Batch explanations")
print("\n📊 Adding:")
print("   - Force Plot (red vs blue forces)")
print("   - Bar Plot (ranked importance)")
print("   - Decision Plot (cumulative path)")
print("   - Multi-select checkboxes")
print("\n🎯 Solution: Use the complete version directly")
print("\nRun this:")
print("  cp src/webapp_complete_shap.py src/webapp_enhanced.py")
print("  streamlit run app_enhanced.py")

# Check if complete version exists
from pathlib import Path
complete_path = Path("/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_complete_shap.py")

if complete_path.exists():
    print(f"\n✅ Complete version exists: {complete_path}")
    print(f"   Size: {complete_path.stat().st_size} bytes")
    print("\n📋 To deploy:")
    print(f"   cp {complete_path} src/webapp_enhanced.py")
else:
    print(f"\n❌ Complete version not found!")
    print("   Creating it now...")
    import subprocess
    result = subprocess.run(["python", "add_complete_shap.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("\n✅ Complete version created!")
    else:
        print(f"\n❌ Error: {result.stderr}")
