# PSI MONITORING - INTEGRATION INSTRUCTIONS

## ✅ COMPLETED CHANGES

### 1. Session State (Lines 83-91) - DONE ✅
Added to session state:
```python
if 'psi_monitor' not in st.session_state:
    st.session_state.psi_monitor = None
if 'psi_results' not in st.session_state:
    st.session_state.psi_results = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
```

## 🔧 REMAINING CHANGES NEEDED

### 2. Update load_data function (Around line 114)

**FIND:**
```python
@st.cache_data
def load_data(config):
    """Load training/validation data."""
    try:
        data_ingestion = DataIngestion(config)
        train_df, val_df, test_df = data_ingestion.load_train_val_test()
        return train_df, val_df, test_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None
```

**REPLACE WITH:**
```python
@st.cache_data
def load_data(config):
    """Load training/validation data."""
    try:
        data_ingestion = DataIngestion(config)
        train_df, val_df, test_df = data_ingestion.load_train_val_test()
        return train_df, val_df, test_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None
```
(No change needed - already returns all three)

### 3. Update Data Loading in Sidebar (Around line 322)

**FIND:**
```python
if st.session_state.config and st.button("📁 Load Training Data"):
    train_df, val_df, test_df = load_data(st.session_state.config)
    if val_df is not None:
        st.session_state.data = val_df
        st.success(f"✅ Loaded {len(val_df)} claims")
```

**REPLACE WITH:**
```python
if st.session_state.config and st.button("📁 Load Training Data"):
    train_df, val_df, test_df = load_data(st.session_state.config)
    if val_df is not None:
        st.session_state.train_data = train_df  # Store for PSI
        st.session_state.data = val_df  # Validation data
        st.session_state.test_data = test_df  # Store for PSI
        st.success(f"✅ Loaded {len(val_df)} validation claims")
        st.info(f"📊 Training: {len(train_df)}, Test: {len(test_df)}")
```

### 4. Update Tabs List (Around line 360)

**FIND:**
```python
# Main content tabs - conditionally add SHAP
tabs_list = ["📊 Dashboard", "🚨 Top Frauds", "📈 Feature Importance", "🔍 Individual Analysis"]
if SHAP_AVAILABLE:
    tabs_list.append("🔬 SHAP Explanations")
tabs_list.append("📁 Export")

tab_objects = st.tabs(tabs_list)
tab1, tab2, tab3, tab4 = tab_objects[0], tab_objects[1], tab_objects[2], tab_objects[3]
if SHAP_AVAILABLE:
    tab_shap, tab5 = tab_objects[4], tab_objects[5]
else:
    tab5 = tab_objects[4]
```

**REPLACE WITH:**
```python
# Main content tabs - conditionally add SHAP and PSI
tabs_list = ["📊 Dashboard", "🚨 Top Frauds", "📈 Feature Importance", "🔍 Individual Analysis"]
if SHAP_AVAILABLE:
    tabs_list.append("🔬 SHAP Explanations")
tabs_list.append("📊 Model Monitoring")  # Add PSI tab
tabs_list.append("📁 Export")

tab_objects = st.tabs(tabs_list)
tab1, tab2, tab3, tab4 = tab_objects[0], tab_objects[1], tab_objects[2], tab_objects[3]

# Handle variable tab assignment based on SHAP availability
if SHAP_AVAILABLE:
    tab_shap = tab_objects[4]
    tab_monitoring = tab_objects[5]
    tab_export = tab_objects[6]
else:
    tab_monitoring = tab_objects[4]
    tab_export = tab_objects[5]
```

### 5. Update Export Tab Reference

**FIND (last tab - around line 1096):**
```python
# Tab 5: Export
with tab5:
```

**REPLACE WITH:**
```python
# Tab: Export
with tab_export:
```

### 6. Insert PSI Monitoring Tab Code

**INSERT BEFORE the Export tab (around line 1095):**

Copy the entire content from `PSI_TAB_CODE.py` file

## 📁 FILES PROVIDED

1. **PSI_TAB_CODE.py** - Complete PSI monitoring tab code (ready to insert)
2. **This file** - Integration instructions

## 🎯 IMPLEMENTATION STEPS

1. ✅ Session state already added
2. Update data loading in sidebar (change #3 above)
3. Update tabs list (change #4 above)  
4. Update export tab reference (change #5 above)
5. Insert PSI tab code (from PSI_TAB_CODE.py)

## ⚡ QUICK IMPLEMENTATION

Run these edits in order:

**Edit 1 - Data Loading:**
Find line ~322, add train/test storage

**Edit 2 - Tabs List:**
Find line ~360, update tabs_list and variable assignments

**Edit 3 - Export Tab:**
Find line ~1096, change `tab5` to `tab_export`

**Edit 4 - Insert PSI Tab:**
Before the Export tab, insert code from PSI_TAB_CODE.py

## 🧪 TESTING

After implementation:
1. Restart Streamlit
2. Load model and training data
3. Navigate to "📊 Model Monitoring" tab
4. Click "🔍 Analyze Data Drift"
5. View PSI scores and drift analysis

## 📊 EXPECTED RESULT

New tab with:
- ✅ Overall PSI metrics
- ✅ Drift status (Stable/Minor/Major)
- ✅ Per-feature PSI bar chart (color-coded)
- ✅ Distribution comparison plots
- ✅ Actionable recommendations
- ✅ CSV download of results

---

**Status:** Ready for implementation
**Estimated Time:** 10-15 minutes
**Complexity:** Medium (multiple file edits)
