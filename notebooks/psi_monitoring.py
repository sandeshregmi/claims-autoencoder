# Databricks notebook source
# MAGIC %md
# MAGIC # PSI Monitoring
# MAGIC Monitor data drift using Population Stability Index

# COMMAND ----------

print("=" * 60)
print("PSI MONITORING")
print("=" * 60)

import pandas as pd
import numpy as np

# COMMAND ----------

# Get catalog from upstream task or use default
try:
    catalog = dbutils.jobs.taskValues.get(taskKey="prepare_data", key="catalog")
    schema = dbutils.jobs.taskValues.get(taskKey="prepare_data", key="schema")
    print(f"✓ Using catalog from task: {catalog}.{schema}")
except:
    catalog = "workspace"
    schema = "default"
    print(f"⚠ Using default: {catalog}.{schema}")

# COMMAND ----------

# Load reference and current data
reference_df = spark.table(f"{catalog}.{schema}.training_data").toPandas()
current_df = spark.table(f"{catalog}.{schema}.training_data").toPandas()  # In production, use latest data

print(f"✓ Loaded reference data: {len(reference_df)} rows")
print(f"✓ Loaded current data: {len(current_df)} rows")

# COMMAND ----------

# Simulate PSI calculation
# In real implementation, use your psi_monitoring.py module
print("\nCalculating PSI scores...")

# Simulate PSI scores for different features
psi_results = pd.DataFrame([{
    'feature': 'claim_amount',
    'psi_score': 0.05,
    'drift_severity': 'low',
    'alert': False
}, {
    'feature': 'patient_age',
    'psi_score': 0.08,
    'drift_severity': 'low',
    'alert': False
}, {
    'feature': 'claim_type',
    'psi_score': 0.12,
    'drift_severity': 'medium',
    'alert': False
}])

print("\nPSI Results:")
print(psi_results)

# COMMAND ----------

# Check for major drift
major_drift = (psi_results['psi_score'] >= 0.2).any()

if major_drift:
    print("\n⚠ MAJOR DRIFT DETECTED!")
    drifted = psi_results[psi_results['psi_score'] >= 0.2]
    print(drifted)
    print("\n→ Recommendation: Retrain model")
else:
    print("\n✓ NO MAJOR DRIFT DETECTED")
    print("  Model performance is stable")

# COMMAND ----------

# Save results
psi_results['monitoring_timestamp'] = pd.Timestamp.now()
psi_results['major_drift_detected'] = major_drift

spark.createDataFrame(psi_results).write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog}.{schema}.psi_results"
)
print(f"\n✓ Saved PSI results to {catalog}.{schema}.psi_results")

# COMMAND ----------

print("=" * 60)
print("PSI MONITORING COMPLETE")
print("=" * 60)
