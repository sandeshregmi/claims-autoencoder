# ✅ SIMPLIFIED - ONE SCRIPT ONLY

## 🎯 The ONLY Script You Need

I've created **one master script** that does everything:

### **start.sh** ← Use this one only!

```bash
chmod +x start.sh
./start.sh
```

## 🗑️ Ignore All Other Scripts

You can ignore these (they were created during development):
- ❌ run_clean_workflow.sh
- ❌ run_clean.sh
- ❌ run_app_direct.sh
- ❌ fix_numpy.sh
- ❌ quick_fix_dependencies.sh
- ❌ clean_workflow.sh
- ❌ make_executable.sh

**Just use `start.sh` - it does everything!**

## 🚀 Complete Workflow

### Step 1: Make it executable (one time only)
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
chmod +x start.sh
```

### Step 2: Run it
```bash
./start.sh
```

### Step 3: Access dashboard
Open browser to: **http://localhost:8501**

### Step 4: Stop it
Press `Ctrl+C` in terminal

## ✨ What start.sh Does

1. ✅ Creates virtual environment (if needed)
2. ✅ Activates virtual environment
3. ✅ Fixes NumPy compatibility automatically
4. ✅ Installs all dependencies
5. ✅ Verifies data and config files
6. ✅ Sets Python path correctly
7. ✅ Suppresses annoying warnings
8. ✅ Launches the dashboard

## 📋 Checklist

Before running, make sure you have:
- [ ] Data file: `data/claims_train.parquet`
- [ ] Config file: `config/starter_config.yaml`

That's it!

## 🎉 Summary

**Old way (confusing):**
- Multiple scripts
- Manual NumPy fixes
- Separate dependency installs
- Complex troubleshooting

**New way (simple):**
```bash
./start.sh
```

Done! 🚀

---

**Just remember: `./start.sh` is all you need!**
