# Databricks notebook source
# MAGIC %md
# MAGIC # Register Model
# MAGIC Register best model if fairness and PSI checks pass

# COMMAND ----------

print("=" * 60)
print("MODEL REGISTRATION")
print("=" * 60)

import pandas as pd

# COMMAND ----------

# Get catalog from upstream task
try:
    catalog = dbutils.jobs.taskValues.get(taskKey="prepare_data", key="catalog")
    schema = dbutils.jobs.taskValues.get(taskKey="prepare_data", key="schema")
    print(f"✓ Using catalog: {catalog}.{schema}")
except:
    catalog = "workspace"
    schema = "default"
    print(f"⚠ Using default: {catalog}.{schema}")

# COMMAND ----------

# Load validation results
eval_df = spark.table(f"{catalog}.{schema}.model_evaluation").toPandas()
fairness_df = spark.table(f"{catalog}.{schema}.fairness_results").toPandas()
psi_df = spark.table(f"{catalog}.{schema}.psi_results").toPandas()

best_model = eval_df['best_model'].iloc[0]
bias_detected = fairness_df['bias_detected'].iloc[0]
major_drift = psi_df['major_drift_detected'].iloc[0]

print(f"Best model: {best_model}")
print(f"Bias detected: {bias_detected}")
print(f"Major drift: {major_drift}")

# COMMAND ----------

# Check if model passes all validations
can_register = not bias_detected and not major_drift

if can_register:
    print("\n✓ ALL VALIDATIONS PASSED!")
    print(f"  → Registering {best_model} model")
    
    # Simulate model registration
    # In production, use MLflow model registry
    registration_result = pd.DataFrame([{
        'model_type': best_model,
        'status': 'registered',
        'fairness_validated': True,
        'drift_validated': True,
        'registered_at': pd.Timestamp.now()
    }])
    
    spark.createDataFrame(registration_result).write.format("delta").mode("append").saveAsTable(
        f"{catalog}.{schema}.model_registry"
    )
    print(f"  ✓ Model registered successfully!")
    
else:
    print("\n⚠ VALIDATION FAILED - Model NOT registered")
    if bias_detected:
        print("  ✗ Bias detected in fairness analysis")
    if major_drift:
        print("  ✗ Major drift detected in PSI monitoring")
    print("\n  → Action required: Review and fix issues before deployment")

# COMMAND ----------

print("=" * 60)
print("MODEL REGISTRATION COMPLETE")
print("=" * 60)
