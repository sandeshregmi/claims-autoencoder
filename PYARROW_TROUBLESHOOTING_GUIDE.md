# Quick Reference: PyArrow Type Conversion Errors in Streamlit

## Common Error Pattern
```
pyarrow.lib.ArrowInvalid: ("Could not convert 'X' with type str: tried to convert to double", 
'Conversion failed for column Y with type object')
```

## Quick Fixes

### Fix 1: Convert to String (Most Common)
```python
# Before st.dataframe() or st.table()
df['problematic_column'] = df['problematic_column'].astype(str)
```

### Fix 2: Convert to Numeric (if all should be numbers)
```python
# Convert to numeric, NaN for non-numeric values
df['problematic_column'] = pd.to_numeric(df['problematic_column'], errors='coerce')
```

### Fix 3: Use Alternative Display Method
```python
# Instead of st.table()
st.write(df)  # More forgiving with mixed types
```

### Fix 4: Split into Multiple Columns
```python
# Separate numeric and categorical data
df['col_numeric'] = pd.to_numeric(df['col'], errors='coerce')
df['col_category'] = df['col'].where(df['col_numeric'].isna(), '')
```

## Diagnostic Commands

```python
# Check data types
print(df.dtypes)

# Check unique values (to find problematic entries)
print(df['column_name'].unique())

# Check for mixed types
print(df['column_name'].apply(type).unique())

# Find rows with non-numeric values in numeric column
print(df[~df['column_name'].apply(lambda x: isinstance(x, (int, float)))])
```

## Prevention Checklist

- [ ] Use consistent data types within columns
- [ ] Convert types explicitly when creating DataFrames
- [ ] Format values as strings if displaying mixed content
- [ ] Test with edge cases (NaN, None, empty strings)
- [ ] Consider using `st.dataframe()` instead of `st.table()` for mixed types

## When This Happens

1. **Identify the column**: Error message tells you which column
2. **Check the data**: Look at `df.dtypes` and `df['col'].unique()`
3. **Decide on type**: String, numeric, or separate columns?
4. **Apply conversion**: Use `.astype()` or `pd.to_numeric()`
5. **Test**: Verify the display works

## Example: Gender Field Issue

```python
# Problem: Gender column has 'M', 'F', and numeric codes like 1, 2
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'gender': ['F', 'M', 1]  # Mixed types!
})

# Solution: Convert all to string
df['gender'] = df['gender'].astype(str)
# Now: ['F', 'M', '1']
```

## Remember

- PyArrow needs consistent types for efficient serialization
- Object dtype = PyArrow will try to infer the type
- Explicit is better than implicit - always specify types!
- When in doubt, convert to string for display purposes
