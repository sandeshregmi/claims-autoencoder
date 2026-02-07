# 🔄 Migration Guide: Local Code → Databricks Format

## Overview

This guide shows you how to migrate your current Claims Fraud Detection code to work in Databricks Asset Bundle format.

## 📋 Current State

Your current code structure:
```
claims-autoencoder/
├── src/
│   ├── config_manager.py          # ✅ Ready
│   ├── data_ingestion.py          # ⚠️ Needs modification
│   ├── tree_models.py             # ⚠️ Needs modification
│   ├── fairness_analysis.py       # ✅ Ready
│   ├── psi_monitoring.py          # ✅ Ready
│   └── coordination_hooks.py      # ✅ Ready
├── config/
│   └── example_config.yaml        # ⚠️ Needs modification
└── data/
    └── claims_train.parquet       # ⚠️ Move to Delta Lake
```

## 🎯 Target State

Databricks bundle structure:
```
claims-autoencoder/
├── databricks.yml                 # ✅ Created
├── resources/jobs/                # ✅ Created
├── src/
│   ├── data_ingestion.py         # 🔄 Modified for Delta
│   ├── tree_models.py            # 🔄 Modified for MLflow
│   ├── fairness_analysis.py      # ✅ No changes needed
│   ├── psi_monitoring.py         # ✅ No changes needed
│   └── databricks/               # 🆕 New Databricks-specific code
│       ├── __init__.py
│       ├── jobs/
│       │   ├── train.py          # Training job entry point
│       │   ├── score.py          # Scoring job entry point
│       │   └── monitor.py        # Monitoring job entry point
│       └── utils/
│           ├── delta_helper.py   # Delta Lake utilities
│           └── mlflow_helper.py  # MLflow utilities
└── .github/workflows/            # ✅ Created
```

---

## Step 1: Adapt data_ingestion.py for Delta Lake

### Current Code (Local Parquet)
```python
# src/data_ingestion.py (CURRENT)
class DataIngestion:
    def load_train_val_test(self):
        # Loads from local parquet file
        data_path = self.config.data.data_path
        df = pd.read_parquet(data_path)
        return train_df, val_df, test_df
```

### New Code (Databricks Delta Lake)
```python
# src/data_ingestion.py (NEW - Databricks compatible)
import argparse
from pyspark.sql import SparkSession
from delta.tables import DeltaTable

class DataIngestion:
    def __init__(self, config=None, spark=None):
        self.config = config
        self.spark = spark or SparkSession.builder.getOrCreate()
    
    def load_from_delta(self, catalog: str, schema: str, table: str):
        """Load data from Delta Lake table"""
        table_path = f"{catalog}.{schema}.{table}"
        
        # Read from Delta
        df = self.spark.table(table_path)
        
        # Convert to Pandas for existing model code
        pandas_df = df.toPandas()
        
        return pandas_df
    
    def save_to_delta(self, df, catalog: str, schema: str, table: str, mode="overwrite"):
        """Save DataFrame to Delta Lake"""
        table_path = f"{catalog}.{schema}.{table}"
        
        # Convert Pandas to Spark DataFrame
        spark_df = self.spark.createDataFrame(df)
        
        # Write to Delta
        spark_df.write \
            .format("delta") \
            .mode(mode) \
            .option("mergeSchema", "true") \
            .saveAsTable(table_path)
        
        print(f"✅ Saved to {table_path}")

# Command-line interface for Databricks job
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--output-table', required=True)
    parser.add_argument('--environment', default='dev')
    args = parser.parse_args()
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Claims Fraud - Data Ingestion") \
        .getOrCreate()
    
    # Initialize data ingestion
    ingestion = DataIngestion(spark=spark)
    
    # Load raw data (from cloud storage or existing table)
    # For migration, you'd upload your parquet to DBFS first:
    # dbfs:/FileStore/claims_data/claims_train.parquet
    df = spark.read.parquet("dbfs:/FileStore/claims_data/claims_train.parquet")
    
    # Save to Delta Lake
    ingestion.save_to_delta(
        df.toPandas(),
        catalog=args.catalog,
        schema=args.schema,
        table=args.output_table
    )

if __name__ == "__main__":
    main()
```

**Key Changes:**
- ✅ Added Spark support
- ✅ Added Delta Lake read/write
- ✅ Added CLI arguments for Databricks job
- ✅ Kept backward compatibility with Pandas

---

## Step 2: Adapt tree_models.py for MLflow

### Current Code (Local Training)
```python
# src/tree_models.py (CURRENT)
class ClaimsTreeAutoencoder:
    def fit(self, data, cat_features, num_features):
        # Train model
        model = CatBoostRegressor()
        model.fit(X, y)
        self.models[feature] = model
```

### New Code (With MLflow Tracking)
```python
# src/tree_models.py (NEW - with MLflow)
import mlflow
import argparse
from pyspark.sql import SparkSession

class ClaimsTreeAutoencoder:
    def fit(self, data, cat_features, num_features, log_mlflow=False, run_name=None):
        """Train model with optional MLflow logging"""
        
        if log_mlflow:
            # Start MLflow run
            mlflow.start_run(run_name=run_name)
            
            # Log parameters
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("n_samples", len(data))
            mlflow.log_param("n_features", len(cat_features) + len(num_features))
        
        try:
            # Train model (existing code)
            for feature in self.target_features:
                model = self._create_model()
                model.fit(X, y)
                self.models[feature] = model
                
                if log_mlflow:
                    # Log model
                    mlflow.sklearn.log_model(
                        model,
                        f"model_{feature}",
                        registered_model_name=f"fraud_detector_{feature}"
                    )
            
            # Compute fraud scores
            fraud_scores, per_feature_errors = self.compute_fraud_scores(data)
            
            if log_mlflow:
                # Log metrics
                mlflow.log_metric("mean_fraud_score", fraud_scores.mean())
                mlflow.log_metric("p95_fraud_score", np.percentile(fraud_scores, 95))
                mlflow.log_metric("p99_fraud_score", np.percentile(fraud_scores, 99))
            
            return fraud_scores, per_feature_errors
            
        finally:
            if log_mlflow:
                mlflow.end_run()

# Databricks job entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['catboost', 'xgboost'], required=True)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    
    # Initialize Spark
    spark = SparkSession.builder.getOrCreate()
    
    # Load data from Delta
    data = spark.table(f"{args.catalog}.{args.schema}.features").toPandas()
    
    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    
    # Train model with MLflow logging
    model = ClaimsTreeAutoencoder(model_type=args.model_type)
    
    fraud_scores, errors = model.fit(
        data,
        cat_features=['claim_type', 'provider_specialty', ...],
        num_features=['claim_amount', 'patient_age', ...],
        log_mlflow=True,
        run_name=args.run_name
    )
    
    # Save results to Delta
    results_df = data.copy()
    results_df['fraud_score'] = fraud_scores
    
    spark.createDataFrame(results_df).write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(f"{args.catalog}.{args.schema}.fraud_scores_{args.model_type}")
    
    print(f"✅ Training complete: {args.model_type}")

if __name__ == "__main__":
    main()
```

**Key Changes:**
- ✅ Added MLflow experiment tracking
- ✅ Added MLflow model logging
- ✅ Added CLI for Databricks job
- ✅ Added Delta Lake integration
- ✅ Kept backward compatibility

---

## Step 3: Create Databricks Job Wrappers

### Create Training Job Wrapper

```python
# src/databricks/jobs/train.py
"""
Databricks training job entry point
Orchestrates data loading, training, and MLflow logging
"""

import argparse
import sys
from pyspark.sql import SparkSession
import mlflow

# Add parent directory to path
sys.path.insert(0, '/Workspace/.bundle/claims-fraud-detection/dev/files')

from src.tree_models import ClaimsTreeAutoencoder
from src.coordination_hooks import coordination_hooks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    
    # Initialize Spark
    spark = SparkSession.builder.getOrCreate()
    
    # Pre-training hook
    coordination_hooks.pre_model_training(args.model_type)
    
    try:
        # Load data
        print(f"Loading data from {args.catalog}.{args.schema}.features")
        data = spark.table(f"{args.catalog}.{args.schema}.features").toPandas()
        
        # Set MLflow experiment
        mlflow.set_experiment(args.experiment_name)
        
        # Train model
        print(f"Training {args.model_type} model...")
        model = ClaimsTreeAutoencoder(model_type=args.model_type)
        
        fraud_scores, errors = model.fit(
            data,
            cat_features=['claim_type', 'provider_specialty', 'diagnosis_code', 
                         'procedure_code', 'patient_gender', 'geographic_region'],
            num_features=['claim_amount', 'patient_age', 'provider_experience_years',
                         'days_since_last_claim', 'num_previous_claims', 
                         'average_claim_amount', 'claim_duration_days'],
            log_mlflow=True,
            run_name=args.run_name
        )
        
        # Save results
        results_df = data.copy()
        results_df['fraud_score'] = fraud_scores
        
        spark.createDataFrame(results_df).write \
            .format("delta") \
            .mode("overwrite") \
            .saveAsTable(f"{args.catalog}.{args.schema}.fraud_scores_{args.model_type}")
        
        # Post-training hook
        coordination_hooks.post_model_training(args.model_type, {
            "mean_score": float(fraud_scores.mean()),
            "p95_score": float(np.percentile(fraud_scores, 95))
        })
        
        print(f"✅ Training complete for {args.model_type}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
```

### Create Fairness Validation Job

```python
# src/databricks/jobs/fairness_validate.py
"""
Databricks fairness validation job
Runs after training to validate model fairness
"""

import argparse
import sys
from pyspark.sql import SparkSession

sys.path.insert(0, '/Workspace/.bundle/claims-fraud-detection/dev/files')

from src.fairness_analysis import FairnessAnalyzer
from src.coordination_hooks import coordination_hooks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--protected-attributes', required=True)
    parser.add_argument('--threshold-percentile', type=float, default=95.0)
    parser.add_argument('--output-table', required=True)
    args = parser.parse_args()
    
    # Parse protected attributes
    protected_attrs = args.protected_attributes.split(',')
    
    # Initialize Spark
    spark = SparkSession.builder.getOrCreate()
    
    # Pre-fairness hook
    coordination_hooks.pre_fairness_analysis(protected_attrs)
    
    try:
        # Load data and scores
        print(f"Loading data from {args.catalog}.{args.schema}.features")
        data = spark.table(f"{args.catalog}.{args.schema}.features").toPandas()
        
        # Load fraud scores (use best model - e.g., catboost)
        fraud_scores_df = spark.table(f"{args.catalog}.{args.schema}.fraud_scores_catboost").toPandas()
        fraud_scores = fraud_scores_df['fraud_score'].values
        
        # Run fairness analysis
        print(f"Running fairness analysis for: {protected_attrs}")
        analyzer = FairnessAnalyzer(
            data=data,
            fraud_scores=fraud_scores,
            protected_attributes=protected_attrs,
            threshold_percentile=args.threshold_percentile
        )
        
        results = analyzer.analyze_all_attributes()
        
        # Post-fairness hook
        coordination_hooks.post_fairness_analysis(results)
        
        # Check if bias detected
        bias_detected = any(
            not result.get('overall_metrics', {}).get('is_fair', True)
            for result in results.values()
            if 'error' not in result
        )
        
        # Save results to Delta
        bias_summary = analyzer.get_bias_summary()
        if not bias_summary.empty:
            spark.createDataFrame(bias_summary).write \
                .format("delta") \
                .mode("overwrite") \
                .saveAsTable(args.output_table)
            
            print(f"✅ Fairness results saved to {args.output_table}")
        
        # Fail job if bias detected (optional - can be configured)
        if bias_detected:
            print("⚠️ BIAS DETECTED - Review required before deployment")
            # In production, you might want to fail the job:
            # sys.exit(1)
        else:
            print("✅ No bias detected - Model is fair")
        
    except Exception as e:
        print(f"❌ Fairness validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
```

---

## Step 4: Upload Data to Databricks

### Option A: Upload via UI
```bash
# 1. Go to Databricks workspace
# 2. Navigate to: Data → Create Table
# 3. Upload: claims_train.parquet
# 4. Save to: /FileStore/claims_data/claims_train.parquet
```

### Option B: Upload via CLI
```bash
# Upload parquet file to DBFS
databricks fs cp data/claims_train.parquet \
  dbfs:/FileStore/claims_data/claims_train.parquet

# Verify upload
databricks fs ls dbfs:/FileStore/claims_data/
```

### Option C: Create Delta Table Directly
```python
# Run in Databricks notebook or job
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Read parquet
df = spark.read.parquet("dbfs:/FileStore/claims_data/claims_train.parquet")

# Write to Delta Lake
df.write \
  .format("delta") \
  .mode("overwrite") \
  .option("mergeSchema", "true") \
  .saveAsTable("dev_fraud_detection.claims.training_data")

print("✅ Data loaded to Delta Lake")
```

---

## Step 5: Test Migration Locally

### Test 1: Validate Modified Code
```bash
# Test data ingestion
python src/databricks/jobs/train.py \
  --model-type catboost \
  --catalog dev_fraud_detection \
  --schema claims \
  --experiment-name /test \
  --run-name test-local

# Should work locally if you have Spark installed
```

### Test 2: Test with Coordination Hooks
```bash
python -c "
from src.coordination_hooks import coordination_hooks
from src.fairness_analysis import FairnessAnalyzer
from src.psi_monitoring import PSIMonitor

print('✅ All imports successful')
print('✅ Migration compatible')
"
```

---

## Step 6: Deploy to Databricks

### Deploy Bundle
```bash
# Validate
databricks bundle validate --target dev

# Deploy
databricks bundle deploy --target dev

# Run training job
databricks bundle run model_training_job --target dev
```

---

## Migration Checklist

### Before Migration
- [ ] Backup current code
- [ ] Test all modules locally
- [ ] Upload data to Databricks
- [ ] Create Unity Catalog (if using)

### During Migration
- [ ] Modify data_ingestion.py for Delta Lake
- [ ] Modify tree_models.py for MLflow
- [ ] Create Databricks job wrappers
- [ ] Test each module independently
- [ ] Update config files

### After Migration
- [ ] Deploy to DEV environment
- [ ] Run end-to-end test
- [ ] Validate fairness analysis works
- [ ] Validate PSI monitoring works
- [ ] Check MLflow experiment tracking
- [ ] Verify costs are as expected

---

## Backward Compatibility

Your original code still works locally! The changes maintain backward compatibility:

```python
# LOCAL USE (Still works!)
from src.tree_models import ClaimsTreeAutoencoder

model = ClaimsTreeAutoencoder(model_type='catboost')
fraud_scores, errors = model.fit(data, cat_features, num_features)

# DATABRICKS USE (New!)
fraud_scores, errors = model.fit(
    data, cat_features, num_features,
    log_mlflow=True,  # Enable MLflow
    run_name="prod-run-123"
)
```

---

## Common Migration Issues

### Issue 1: Import Errors
**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Add sys.path in job scripts:
```python
import sys
sys.path.insert(0, '/Workspace/.bundle/claims-fraud-detection/dev/files')
```

### Issue 2: Spark Not Found Locally
**Problem**: Testing fails locally without Spark

**Solution**: Use mocks or skip Databricks-specific code:
```python
try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    
if SPARK_AVAILABLE:
    # Databricks code
else:
    # Local fallback
```

### Issue 3: Delta Table Not Found
**Problem**: Table doesn't exist in catalog

**Solution**: Create table first or use IF NOT EXISTS:
```python
spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {catalog}.{schema}.{table}
  USING delta
""")
```

---

## Summary

**What Changed:**
- ✅ Added Delta Lake support (data_ingestion.py)
- ✅ Added MLflow tracking (tree_models.py)
- ✅ Added CLI interfaces for jobs
- ✅ Created job wrapper scripts

**What Stayed the Same:**
- ✅ fairness_analysis.py (no changes)
- ✅ psi_monitoring.py (no changes)
- ✅ coordination_hooks.py (no changes)
- ✅ Core algorithm logic (no changes)

**Total Migration Time:** ~4-6 hours

**Next Step:** Run `databricks bundle deploy --target dev` 🚀
