# What Does start.sh Do?

## Yes, start.sh is ONLY for the Dashboard

`./start.sh` does one thing: **Launches the Streamlit web dashboard**

## What's in the Dashboard?

The dashboard includes ALL the functionality:

### 1. 📊 Overview Tab
- View fraud statistics
- Model performance metrics
- Data visualizations

### 2. 🔮 Predictions Tab
- **Score individual claims** for fraud risk
- **Batch predictions** on new data
- Real-time fraud detection

### 3. ⭐ Feature Importance Tab
- See which features matter most
- Global importance rankings
- Feature interactions

### 4. 📈 PSI Monitoring Tab
- **Monitor data drift**
- Detect distribution changes
- Alert on feature shifts

### 5. ⚖️ Fairness Analysis Tab
- **Check for bias** in predictions
- Demographic parity analysis
- Equal opportunity metrics

### 6. 🔬 SHAP Analysis Tab
- **Explain predictions** with SHAP values
- Force plots for individual claims
- Summary plots across dataset
- Feature dependence plots

### 7. 🎓 Training Tab (Interactive)
- **Train new models** from the dashboard
- Select model type (XGBoost, CatBoost)
- Tune hyperparameters
- View training progress
- Compare model performance

## So Everything is in the Dashboard?

**YES!** The dashboard is your complete end-to-end pipeline:

```
Dashboard includes:
├── Data exploration ✅
├── Model training ✅
├── Predictions ✅
├── Explainability ✅
├── Monitoring ✅
└── Fairness analysis ✅
```

## Do I Need Other Scripts?

**NO!** Everything you need is accessible through the dashboard web interface.

## What If I Want to Train Models via Command Line?

If you prefer command-line training (not the dashboard), you would need separate scripts. But for now, the dashboard does everything.

## What start.sh Actually Does

```bash
./start.sh
↓
Sets up environment
↓
Installs dependencies
↓
Launches Streamlit Dashboard at http://localhost:8501
↓
You interact with EVERYTHING through the web interface
```

## Summary

- **start.sh** = Launches the web dashboard
- **The dashboard** = Your complete ML pipeline (train, predict, explain, monitor)
- **No other scripts needed** = Everything is in the web UI

**Just run `./start.sh` and do everything from your browser!** 🚀
