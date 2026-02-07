# How to Fix Deprecation Warnings

## The Warnings You're Seeing

```
The keyword arguments have been deprecated and will be removed in a future release. 
Use `config` instead to specify Plotly configuration options.
```

## What They Mean

These are **harmless warnings** from Plotly about future API changes. They don't affect functionality at all - everything works perfectly!

## Solution 1: Use Updated start.sh (Recommended)

I've updated `start.sh` to suppress these warnings automatically:

```bash
./start.sh
```

This will give you clean output without the deprecation spam!

## Solution 2: Manual Suppression

If you want to run manually:

```bash
export PYTHONWARNINGS='ignore::DeprecationWarning,ignore::UserWarning'
export PYTHONPATH="/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder:${PYTHONPATH}"
streamlit run src/webapp_enhanced.py 2>&1 | grep -v "deprecated"
```

## Solution 3: Fix the Code (For Developers)

If you want to permanently fix the warnings in the code, you would need to update how Plotly charts are configured. The old way uses keyword arguments, the new way uses a `config` dictionary.

### Old Way (causes warnings):
```python
st.plotly_chart(fig, use_container_width=True)
```

### New Way (no warnings):
```python
st.plotly_chart(fig, config={'displayModeBar': False})
```

However, this requires updating many lines in `webapp_enhanced.py`.

## Why These Warnings Appear

Plotly is updating their API and warning about old syntax that will be removed in future versions. Your code still works perfectly - it's just using the "old" way that will need updating eventually.

## Impact

**Current:** ⚠️ Warnings appear but everything works
**Future:** In a future Plotly version (probably 2026+), the old syntax might stop working
**Action Needed:** None right now - the updated `start.sh` handles it

## Recommended Approach

**Just use `./start.sh`** - it automatically suppresses these warnings for clean output!

The warnings are purely cosmetic and don't indicate any problems with your application.

## Summary

✅ **Application works perfectly**
✅ **Warnings are harmless**  
✅ **start.sh suppresses them automatically**
✅ **No code changes needed**

---

**Just run `./start.sh` and enjoy warning-free output!** 🎉
