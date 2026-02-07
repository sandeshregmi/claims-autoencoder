# Claims Fraud Detection - Quick Start

## 🚀 One Command to Run Everything

```bash
chmod +x start.sh
./start.sh
```

That's it! This single script will:
1. ✅ Create virtual environment (if needed)
2. ✅ Install all dependencies
3. ✅ Fix NumPy compatibility
4. ✅ Verify data and config
5. ✅ Launch the dashboard

## 📍 Access Your Dashboard

After running `./start.sh`, open your browser to:
**http://localhost:8501**

## 🛑 Stop the Application

Press `Ctrl+C` in the terminal

## 📁 What You Need

Make sure you have:
- `data/claims_train.parquet` - Your training data
- `config/starter_config.yaml` - Configuration file

## ❓ Troubleshooting

### Permission Denied
```bash
chmod +x start.sh
```

### Port Already in Use
The script will tell you if port 8501 is busy. Stop other Streamlit apps first.

### Missing Data File
Place your `claims_train.parquet` file in the `data/` directory.

## 📚 Documentation

For detailed information, see:
- `README_CLEAN.md` - Complete guide
- `START_HERE.md` - Overview

---

**Remember: You only need to run `./start.sh` - ignore all other .sh files!**
