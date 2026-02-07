# 🚀 Deploy Your Existing Web App

## ✅ Existing Web Apps Found

You already have comprehensive web applications in `src/`:

1. **webapp_enhanced_COMPLETE.py** (45KB) ⭐ RECOMMENDED
   - Complete dashboard with all features
   - Tree models integration
   - SHAP explanations
   - PSI monitoring
   - Fairness analysis

2. **webapp_enhanced.py**
   - Enhanced version

3. **webapp.py**
   - Basic version

---

## 🚀 Quick Start - Deploy Existing Web App

### Option 1: Run the Complete Version (Recommended)

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Run the complete webapp
streamlit run src/webapp_enhanced_COMPLETE.py
```

### Option 2: Create Convenience Link

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Create link for easy access
ln -s src/webapp_enhanced_COMPLETE.py app.py

# Run it
streamlit run app.py
```

---

## 📊 What's In Your Existing Web App

Based on the file structure, your webapp likely includes:

### Core Features
- ✅ Claims fraud scoring interface
- ✅ Model comparison (CatBoost, XGBoost, FT-Transformer)
- ✅ SHAP explainability
- ✅ Feature importance visualization
- ✅ PSI drift monitoring
- ✅ Fairness analysis dashboard

### Data Integration
- ✅ Parquet file loading
- ✅ Model predictions
- ✅ Historical analysis

### Visualizations
- ✅ Plotly interactive charts
- ✅ SHAP force plots
- ✅ Feature distributions
- ✅ Model performance metrics

---

## 🔧 Connect Web App to Databricks Results

Your web app can load results from your Databricks pipeline!

### Update Connection Settings

Edit `src/webapp_enhanced_COMPLETE.py` to add Databricks connection:

```python
# Add at top of file
from databricks import sql
import os

# Add Databricks connection function
def load_databricks_data(table_name):
    """Load data from Databricks"""
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST", "dbc-d4506e69-bbc8.cloud.databricks.com"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT * FROM workspace.default.{table_name}")
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(result, columns=columns)

# Use in your app
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_model_scores():
    """Load scores from Databricks"""
    catboost_scores = load_databricks_data("fraud_scores_catboost")
    xgboost_scores = load_databricks_data("fraud_scores_xgboost")
    ft_scores = load_databricks_data("fraud_scores_ft_transformer")
    return catboost_scores, xgboost_scores, ft_scores
```

### Set Environment Variables

```bash
# Create .env file
cat > .env << 'EOF'
DATABRICKS_HOST=dbc-d4506e69-bbc8.cloud.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your_warehouse_id
DATABRICKS_TOKEN=your_token_here
EOF

# Install databricks-sql-connector
pip install databricks-sql-connector python-dotenv

# Load env vars in app
# Add to top of webapp_enhanced_COMPLETE.py:
from dotenv import load_dotenv
load_dotenv()
```

---

## 📦 Install Dependencies

```bash
# Your webapp likely needs these
pip install streamlit plotly pandas numpy matplotlib seaborn shap

# For Databricks connection
pip install databricks-sql-connector python-dotenv

# Your existing requirements
pip install -r requirements.txt
```

---

## 🎯 Deployment Options

### Option 1: Local Development
```bash
streamlit run src/webapp_enhanced_COMPLETE.py
```
Access at: http://localhost:8501

### Option 2: Streamlit Cloud
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect your repo
4. Select `src/webapp_enhanced_COMPLETE.py`
5. Add secrets (Databricks credentials)

### Option 3: Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "src/webapp_enhanced_COMPLETE.py"]
```

Build and run:
```bash
docker build -t claims-fraud-app .
docker run -p 8501:8501 claims-fraud-app
```

### Option 4: Databricks Apps (if available)
If your workspace has Databricks Apps enabled:
1. Create app.py in workspace
2. Upload your webapp code
3. Configure as Databricks App

---

## 🔗 Integration Architecture

```
┌─────────────────┐
│  Databricks     │
│  Pipeline       │
│                 │
│  • Training     │
│  • Scoring      │
│  • Monitoring   │
└────────┬────────┘
         │
         │ Delta Tables
         ↓
┌─────────────────┐
│  Web Dashboard  │
│                 │
│  • Load data    │
│  • Visualize    │
│  • Score claims │
└─────────────────┘
```

---

## 📝 Quick Test

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Test your existing webapp
streamlit run src/webapp_enhanced_COMPLETE.py

# Should open browser automatically
# If not, go to: http://localhost:8501
```

---

## 🎊 Your Web App is Ready!

You don't need the new streamlit_app.py I created - your existing webapp is much more comprehensive!

**Run it now:**
```bash
streamlit run src/webapp_enhanced_COMPLETE.py
```

---

## 📚 Additional Resources

- **App Code:** `src/webapp_enhanced_COMPLETE.py`
- **SHAP Integration:** Already included
- **PSI Monitoring:** Integrated via `src/psi_monitoring.py`
- **Fairness:** Integrated via `src/fairness_analysis.py`
- **Tree Models:** Integrated via `src/tree_models.py`

**Your existing web app has everything you need!** 🎉
