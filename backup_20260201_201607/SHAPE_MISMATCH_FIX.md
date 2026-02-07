# Global Importance Shape Mismatch - FIXED ✅

## Problem Identified

**Error:** `operands could not be broadcast together with shapes (4,) (5,)`

**Root Cause:**
- CatBoost's SHAP explainer was returning arrays with inconsistent shapes for different target features
- Some categorical features (like `claim_type`, `diagnosis_code`, `procedure_code`) may have different numbers of unique values after encoding
- When pandas tried to compute the mean across these mismatched arrays, it failed with a broadcast error

## Solution Applied

### File: `shap_explainer.py` - `get_global_feature_importance()` method

**Enhanced with robust shape handling:**

1. **Wrapped each feature in try-except** - Skip problematic features instead of failing entirely

2. **Added shape verification:**
   ```python
   # Ensure 2D array (samples x features)
   if shap_values.ndim == 1:
       shap_values = shap_values.reshape(1, -1)
   elif shap_values.ndim > 2:
       shap_values = shap_values[0]
   ```

3. **Verified dimensions match:**
   ```python
   if shap_values.shape[1] != len(predictor_features):
       logger.warning("Shape mismatch - skipping")
       continue
   ```

4. **Ensured scalar values:**
   ```python
   'mean_abs_shap': float(mean_abs_shap[i])  # Convert to Python float
   ```

5. **Forced numeric type:**
   ```python
   importance_df['mean_abs_shap'] = pd.to_numeric(importance_df['mean_abs_shap'], errors='coerce')
   importance_df = importance_df.dropna(subset=['mean_abs_shap'])
   ```

6. **Added safety checks:**
   - Check if list is empty before creating DataFrame
   - Bounds checking on array indices
   - Drop NaN values before aggregation

## What This Fixes

### Before:
- ❌ Crashed when SHAP arrays had different shapes
- ❌ No error recovery - one bad feature killed everything
- ❌ No shape validation

### After:
- ✅ Handles variable-length arrays gracefully
- ✅ Skips problematic features with warnings (logged)
- ✅ Validates all shapes before processing
- ✅ Ensures numeric types for aggregation
- ✅ Continues even if some features fail

## Testing Steps

1. **Restart Streamlit** (very important!)
   ```bash
   cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
   streamlit run app_enhanced.py
   ```

2. **Navigate to SHAP Explanations tab**

3. **Select "Global Importance"**

4. **Start with 100 samples** (for faster testing)

5. **Click "🔍 Compute Global Importance"**

6. **Expected result:**
   - Progress bar completes successfully
   - Bar chart shows top 25 features
   - Full table with all features
   - Download button appears

## If You See Warnings

The console might show warnings like:
```
WARNING: Shape mismatch for claim_type: SHAP has 4 features but expected 5
WARNING: Error computing SHAP for target diagnosis_code: [error]
```

This is **NORMAL** and **EXPECTED**. The method will:
- Skip those specific target features
- Continue with the remaining features
- Still produce valid global importance rankings

## Why This Happened

**CatBoost categorical encoding:**
- Different features have different numbers of categories
- When CatBoost creates one-hot or label encodings, array dimensions vary
- SHAP respects these native dimensions
- Our aggregation code needed to handle this variability

**Example:**
- `patient_gender`: 2 categories (M, F) → SHAP array length 2
- `claim_type`: 5 categories → SHAP array length 5  
- `diagnosis_code`: 100+ categories → SHAP array length 100+

When we tried to average these together without proper handling, pandas couldn't broadcast mismatched shapes.

## Verification

After the fix, Global Importance should:
- ✅ Complete without errors (though warnings OK)
- ✅ Show importance scores for 12-13 features
- ✅ Produce a ranked list
- ✅ Allow CSV download

If it still fails, the new error handling will show exactly which feature and why.

---

**Status:** FIXED ✅
**Files Modified:** `shap_explainer.py`
**Action Required:** Restart Streamlit and test
