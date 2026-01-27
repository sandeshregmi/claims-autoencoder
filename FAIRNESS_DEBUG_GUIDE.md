# 🔍 FAIRNESS TAB DEBUGGING GUIDE

## Issue: Fairness Tab Not Generating Results

### Quick Checks:

1. **Test the fairness module:**
```bash
python3 test_fairness.py
```

This will verify the fairness_analysis.py module works independently.

2. **Check Streamlit for errors:**
- Look at the terminal where Streamlit is running
- Check for error messages when clicking "⚖️ Run Fairness Analysis"
- Open browser console (F12) for JavaScript errors

3. **Common Issues:**

#### Issue 1: Module Not Found
**Symptom:** Error: "No module named 'src.fairness_analysis'"

**Solution:**
```bash
# Verify file exists
ls -la src/fairness_analysis.py

# If missing, the file should be at:
# /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/fairness_analysis.py
```

#### Issue 2: No Protected Attributes Found
**Symptom:** Button shows but no results after clicking

**Debug Steps:**
1. Check what columns are in your data:
   - In Streamlit, go to Individual Analysis tab
   - Look at the claim details - what columns exist?

2. The fairness analysis looks for these columns:
   - 'patient_gender', 'gender'
   - 'patient_age', 'age', 'age_group', 'patient_age_group'
   - 'geographic_region', 'region', 'state'
   - 'race', 'ethnicity'
   - 'income_level', 'socioeconomic_status'

3. If your data doesn't have these, it will use categorical columns

**Solution:** Check the multiselect dropdown - are there any options?

#### Issue 3: Data Not Loaded
**Symptom:** Message "👈 Please load data and compute fraud scores first"

**Solution:**
1. Sidebar → Load Model
2. Sidebar → Load Training Data
3. Sidebar → Train Model
4. Sidebar → Compute Fraud Scores
5. Then go to Fairness Analysis tab

#### Issue 4: Import Error in Webapp
**Symptom:** Error about importing FairnessAnalyzer

**Check this in terminal:**
```python
python3 << EOF
import sys
sys.path.insert(0, '.')
from src.fairness_analysis import FairnessAnalyzer
print("✅ Import works!")
EOF
```

### Detailed Debugging:

If the button clicks but nothing happens:

1. **Add debug output to webapp:**

Edit `src/webapp_enhanced.py` and find the fairness tab section. Add prints:

```python
if st.button("⚖️ Run Fairness Analysis", type="primary"):
    st.write("DEBUG: Button clicked!")  # ADD THIS
    st.write(f"DEBUG: Selected attributes: {selected_attributes}")  # ADD THIS
    
    if not selected_attributes:
        st.error("Please select at least one protected attribute")
    else:
        st.write("DEBUG: Starting analysis...")  # ADD THIS
        with st.spinner("Analyzing fairness across groups..."):
            try:
                st.write("DEBUG: Importing FairnessAnalyzer...")  # ADD THIS
                from src.fairness_analysis import FairnessAnalyzer
                
                st.write("DEBUG: Creating analyzer...")  # ADD THIS
                analyzer = FairnessAnalyzer(...)
                
                st.write("DEBUG: Running analysis...")  # ADD THIS
                # ... rest of code
```

2. **Check the actual error:**

Look in the Streamlit terminal for the full traceback.

### Manual Test in Python:

```python
# Test with your actual data
import pandas as pd
import numpy as np
from src.fairness_analysis import FairnessAnalyzer

# Load your data
from src.config_manager import ConfigManager
from src.data_ingestion import DataIngestion

config_manager = ConfigManager('config/example_config.yaml')
config = config_manager.get_config()
data_ingestion = DataIngestion(config)
train_df, val_df, test_df = data_ingestion.load_train_val_test()

# Check columns
print("Available columns:")
print(val_df.columns.tolist())

# Check for protected attributes
protected_attrs = [
    'patient_gender', 'gender',
    'patient_age', 'age', 'age_group', 'patient_age_group',
    'geographic_region', 'region', 'state'
]

found_attrs = [attr for attr in protected_attrs if attr in val_df.columns]
print(f"\nFound protected attributes: {found_attrs}")

# If none found, use categorical
if not found_attrs:
    found_attrs = val_df.select_dtypes(include=['object', 'category']).columns.tolist()[:3]
    print(f"Using categorical columns: {found_attrs}")

# Generate some fake fraud scores for testing
fake_scores = np.random.lognormal(10, 2, len(val_df))

# Try to run analysis
analyzer = FairnessAnalyzer(
    data=val_df,
    fraud_scores=fake_scores,
    protected_attributes=found_attrs,
    threshold_percentile=95.0
)

results = analyzer.analyze_all_attributes()
print(f"\n✅ Analysis worked! Found {len(results)} attributes")

bias_summary = analyzer.get_bias_summary()
print("\nBias Summary:")
print(bias_summary)
```

### Expected Behavior:

When working correctly, you should see:

1. **Configuration section:**
   - Reference Data count
   - Current Data count
   - Number of Bins selector

2. **After clicking button:**
   - "✅ Fairness analysis complete!" message
   - Balloons animation

3. **Results section:**
   - 4 metrics at top (Attributes Analyzed, Fair Attributes, Biased Attributes, Avg Fairness Score)
   - Summary table
   - Dropdown to select attribute
   - Charts and detailed analysis

### Still Not Working?

Run this comprehensive diagnostic:

```bash
python3 << 'EOF'
print("=" * 60)
print("COMPREHENSIVE DIAGNOSTIC")
print("=" * 60)

# 1. Check file exists
import os
if os.path.exists('src/fairness_analysis.py'):
    print("✅ fairness_analysis.py exists")
    print(f"   Size: {os.path.getsize('src/fairness_analysis.py')} bytes")
else:
    print("❌ fairness_analysis.py NOT FOUND")

# 2. Check imports
try:
    from src.fairness_analysis import FairnessAnalyzer
    print("✅ FairnessAnalyzer imports correctly")
except Exception as e:
    print(f"❌ Import failed: {e}")

# 3. Check dependencies
deps = ['numpy', 'pandas', 'scipy']
for dep in deps:
    try:
        __import__(dep)
        print(f"✅ {dep} installed")
    except:
        print(f"❌ {dep} NOT installed")

# 4. Check webapp has fairness tab
try:
    with open('src/webapp_enhanced.py', 'r') as f:
        content = f.read()
        if 'tab_fairness' in content:
            print("✅ Fairness tab code is in webapp")
            count = content.count('tab_fairness')
            print(f"   Found {count} references to tab_fairness")
        else:
            print("❌ Fairness tab code NOT in webapp")
except Exception as e:
    print(f"❌ Could not read webapp: {e}")

print("=" * 60)
EOF
```

Copy the output and I can help debug further!
