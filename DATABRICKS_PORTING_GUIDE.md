# 🚀 Databricks Porting Guide

## ✅ Your Project is Ready for Databricks!

Everything is automated and controlled from `config/config.yaml`. Just update the config and go!

---

## 📋 What's Configured

### 1. Databricks Data Source ✅
```yaml
databricks:
  data_source:
    catalog: "machine_learning"      # ✅ YOUR CATALOG
    schema: "casetracker"            # ✅ YOUR SCHEMA
    tables:
      claims: "claims"               # Your table name
```

### 2. Column Mapping ✅
Maps your Databricks columns to standard feature names:
```yaml
column_mapping:
  claim_amount: "claim_amt"         # Databricks -> Standard
  patient_age: "age"
  # ... etc
```

### 3. SQL Query Template ✅
Auto-generates queries with joins:
```sql
SELECT c.*, p.patient_age, pr.provider_specialty
FROM machine_learning.casetracker.claims c
LEFT JOIN machine_learning.casetracker.patients p ...
```

### 4. Storage Paths ✅
```yaml
storage:
  volume_path: "/Volumes/machine_learning/casetracker/claims_fraud"
```

---

## 🎯 How to Port to Databricks

### Step 1: Update Config (5 minutes)

Edit `config/config.yaml`:

```yaml
databricks:
  data_source:
    catalog: "machine_learning"     # Keep this
    schema: "casetracker"           # Keep this
    
    # ✏️ UPDATE: Your actual table names
    tables:
      claims: "your_claims_table_name"
      patients: "your_patients_table_name"  
      providers: "your_providers_table_name"
    
    # ✏️ UPDATE: Map YOUR Databricks columns to feature names
    column_mapping:
      claim_amount: "your_claim_amount_column"
      patient_age: "your_age_column"
      provider_experience_years: "your_exp_column"
      # ... update all mappings
```

### Step 2: Deploy to Databricks (10 minutes)

```bash
# 1. Upload project to Databricks
databricks workspace import-dir \
  /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder \
  /Workspace/Shared/claims_fraud

# 2. Install package in notebook
%pip install -e /Workspace/Shared/claims_fraud

# 3. Load and run
from claims_fraud.config.manager import load_config
from claims_fraud.data.databricks_loader import load_databricks_data

config = load_config("/Workspace/Shared/claims_fraud/config/config.yaml")
df = load_databricks_data(config)
```

### Step 3: Test Data Loading (2 minutes)

```python
# In Databricks notebook
from claims_fraud.data.databricks_loader import DatabricksDataLoader
from claims_fraud.config.manager import load_config

config = load_config("config/config.yaml")
loader = DatabricksDataLoader(config)

# Load data with date range
df = loader.load_data(
    start_date="2024-01-01",
    end_date="2024-12-31"
)

print(f"✅ Loaded {len(df):,} rows")
print(f"✅ Features: {list(df.columns)}")
```

---

## 🔧 Customization Points

### 1. Custom SQL Query

Edit `query_template` in config:
```yaml
query_template: |
  SELECT 
    c.claim_id,
    c.claim_amount,
    p.age as patient_age,
    pr.specialty as provider_specialty
  FROM {catalog}.{schema}.claims c
  LEFT JOIN {catalog}.{schema}.patients p 
    ON c.patient_id = p.id
  WHERE c.status = 'APPROVED'
    AND c.date >= '{start_date}'
```

### 2. Add Filters

```yaml
filters:
  additional_conditions: "AND claim_status = 'APPROVED' AND amount > 1000"
```

### 3. Adjust Feature Engineering

All in config - no code changes:
```yaml
feature_engineering:
  derived_features:
    - name: high_value_indicator
      formula: "claim_amount > 10000"
      dtype: bool
```

---

## 📊 Data Flow (100% Automated)

```
1. Databricks Unity Catalog
   ↓ (SQL query from config)
2. Raw Data
   ↓ (Column mapping from config)
3. Standardized Features
   ↓ (Validation from config)
4. Clean Data
   ↓ (Feature engineering from config)
5. Processed Features
   ↓ (Business rules from config)
6. Fraud Scores
```

**Everything controlled by config.yaml!**

---

## ✅ Pre-Flight Checklist

Before porting, verify:

- [ ] Databricks workspace access
- [ ] Unity Catalog `machine_learning.casetracker` exists
- [ ] Tables contain required columns
- [ ] Column names match `column_mapping` in config
- [ ] Secrets configured (if needed)
- [ ] MLflow experiment path exists

---

## 🧪 Testing Locally First

Test without Databricks:

```python
# Set data source to 'local' in config
config.data.source = "local"

# Use sample data
from claims_fraud.data.databricks_loader import DatabricksDataLoader
loader = DatabricksDataLoader(config)
df = loader._generate_sample_data(n_samples=1000)
```

---

## 📝 What You Need to Update

### Minimum Changes:
1. **Table names** in `databricks.data_source.tables`
2. **Column mapping** in `databricks.data_source.column_mapping`

### Optional Changes:
3. SQL query template (if complex joins needed)
4. Storage paths (if using different location)
5. Cluster config (if different specs needed)

---

## 🎯 Example: Complete Port

```yaml
# 1. Update config.yaml
databricks:
  data_source:
    catalog: "machine_learning"
    schema: "casetracker"
    tables:
      claims: "claims_2024"          # ← Your table
    column_mapping:
      claim_amount: "total_amount"   # ← Your columns
      patient_age: "patient_age_yrs"
      # ... etc

# 2. Upload to Databricks
# 3. Run in notebook:
```

```python
from claims_fraud.config.manager import load_config
from claims_fraud.data.databricks_loader import load_databricks_data
from claims_fraud.data.validation import DataValidator
from claims_fraud.core.business_rules import BusinessRulesEngine

# Load config
config = load_config("config/config.yaml")

# Load data (auto-queries Unity Catalog)
df = load_databricks_data(config, start_date="2024-01-01")

# Validate (auto-validates against config)
validator = DataValidator(config)
df = validator.auto_fix_data(df)

# Apply business rules (auto-applies from config)
engine = BusinessRulesEngine(config)
df_scored = engine.apply_all_rules(df)

# Done! All automated from config
```

---

## 🚀 That's It!

**Your project is 100% config-driven and Databricks-ready!**

Just update the config and deploy. No code changes needed! 🎉

---

## 📚 Files You'll Use

1. **config/config.yaml** - All settings (UPDATE THIS)
2. **src/claims_fraud/data/databricks_loader.py** - Data loading (uses config)
3. **src/claims_fraud/data/validation.py** - Validation (uses config)
4. **src/claims_fraud/core/business_rules.py** - Business logic (uses config)

**Everything reads from config - nothing hardcoded!** ✅
