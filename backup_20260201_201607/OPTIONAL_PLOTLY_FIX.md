# Optional: Fix Plotly Deprecation Warnings

## Warning Message
```
The keyword arguments have been deprecated and will be removed in a future release. 
Use `config` instead to specify Plotly configuration options.
```

## What Causes This

The deprecated parameter `width="stretch"` in `st.plotly_chart()` calls.

## How to Fix (OPTIONAL)

### Search and Replace in webapp_enhanced.py

**Find all instances of:**
```python
st.plotly_chart(fig, width="stretch", key="some_key")
```

**Replace with:**
```python
st.plotly_chart(fig, use_container_width=True, key="some_key")
```

### Specific Lines to Update

If you want to fix these warnings, update these lines in `/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src/webapp_enhanced.py`:

1. **Line ~430** (Dashboard - Distribution plot):
   ```python
   # BEFORE:
   st.plotly_chart(fig, width="stretch", key="fraud_distribution_main")
   
   # AFTER:
   st.plotly_chart(fig, use_container_width=True, key="fraud_distribution_main")
   ```

2. **Line ~540** (Top Frauds - Anomaly plots):
   ```python
   # BEFORE:
   st.plotly_chart(fig, width="stretch", key=f"anomaly_rank_{i}")
   
   # AFTER:
   st.plotly_chart(fig, use_container_width=True, key=f"anomaly_rank_{i}")
   ```

3. **Line ~570** (Feature Importance):
   ```python
   # BEFORE:
   st.plotly_chart(fig, width="stretch", key="feature_importance_main")
   
   # AFTER:
   st.plotly_chart(fig, use_container_width=True, key="feature_importance_main")
   ```

4. **Line ~590** (Top Claims Heatmap):
   ```python
   # BEFORE:
   st.plotly_chart(fig, width="stretch", key="heatmap_top10")
   
   # AFTER:
   st.plotly_chart(fig, use_container_width=True, key="heatmap_top10")
   ```

5. **Line ~630** (Individual Analysis - Anomalies):
   ```python
   # BEFORE:
   st.plotly_chart(fig, width="stretch", key="individual_analysis_chart")
   
   # AFTER:
   st.plotly_chart(fig, use_container_width=True, key="individual_analysis_chart")
   ```

### Automated Fix Command

You can use this sed command to fix all instances automatically:

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder/src
sed -i.bak 's/width="stretch"/use_container_width=True/g' webapp_enhanced.py
```

This will:
- Replace all `width="stretch"` with `use_container_width=True`
- Create a backup file `webapp_enhanced.py.bak`

## Impact

- **Severity:** LOW (just deprecation warnings)
- **Functionality:** No impact - charts still display correctly
- **Performance:** No impact
- **Priority:** Can be done anytime, not urgent

## Recommendation

Since these are just warnings (not errors), you can:
1. **Fix now** if you want clean logs
2. **Fix later** during regular maintenance
3. **Ignore** until Streamlit actually removes the deprecated parameter

The PyArrow errors we fixed earlier were **critical** and required immediate attention.
These Plotly warnings are **cosmetic** and don't affect functionality.

---

**Note:** The main PyArrow errors have been fixed. This document is for optional cleanup only.
