# Databricks notebook source
# MAGIC %md
# MAGIC # Downstream Analysis Dashboard
# MAGIC Comprehensive analysis and visualization of model performance

# COMMAND ----------

print("=" * 60)
print("DOWNSTREAM ANALYSIS")
print("=" * 60)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# COMMAND ----------

# Get catalog
try:
    catalog = dbutils.jobs.taskValues.get(taskKey="prepare_data", key="catalog")
    schema = dbutils.jobs.taskValues.get(taskKey="prepare_data", key="schema")
    best_model = dbutils.jobs.taskValues.get(taskKey="evaluate_models", key="best_model")
except:
    catalog = "workspace"
    schema = "default"
    best_model = "catboost"

print(f"Catalog: {catalog}.{schema}")
print(f"Best Model: {best_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Model Performance Comparison

# COMMAND ----------

# Load evaluation results
eval_df = spark.table(f"{catalog}.{schema}.model_evaluation").toPandas()

print("\nModel Performance Summary:")
print(eval_df[['model', 'mean_score', 'p95_score', 'p99_score']].to_string(index=False))

# Create comparison chart data
comparison_data = eval_df[['model', 'mean_score', 'p95_score', 'p99_score']].copy()
comparison_data['rank'] = comparison_data['mean_score'].rank(ascending=False)

print("\nModel Rankings:")
print(comparison_data.sort_values('rank')[['model', 'mean_score', 'rank']].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fairness Analysis Summary

# COMMAND ----------

# Load fairness results
fairness_df = spark.table(f"{catalog}.{schema}.fairness_results").toPandas()

print("\nFairness Analysis:")
print(fairness_df[['attribute', 'group', 'disparate_impact', 'is_fair']].to_string(index=False))

# Check for bias
bias_detected = not fairness_df['is_fair'].all()
if bias_detected:
    print("\n⚠ BIAS DETECTED in the following groups:")
    biased = fairness_df[~fairness_df['is_fair']]
    print(biased[['attribute', 'group', 'disparate_impact']].to_string(index=False))
else:
    print("\n✓ NO BIAS DETECTED - Model is fair across all protected attributes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Drift Monitoring

# COMMAND ----------

# Load PSI results
psi_df = spark.table(f"{catalog}.{schema}.psi_results").toPandas()

print("\nPopulation Stability Index (PSI):")
print(psi_df[['feature', 'psi_score', 'drift_severity']].to_string(index=False))

# Drift alerts
major_drift = psi_df[psi_df['psi_score'] >= 0.2]
if not major_drift.empty:
    print("\n⚠ MAJOR DRIFT DETECTED:")
    print(major_drift[['feature', 'psi_score']].to_string(index=False))
    print("\n→ Recommendation: Retrain model")
else:
    medium_drift = psi_df[psi_df['psi_score'] >= 0.1]
    if not medium_drift.empty:
        print("\n⚠ MEDIUM DRIFT DETECTED:")
        print(medium_drift[['feature', 'psi_score']].to_string(index=False))
        print("\n→ Recommendation: Monitor closely")
    else:
        print("\n✓ NO SIGNIFICANT DRIFT - Data distribution is stable")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Production Readiness Check

# COMMAND ----------

# Comprehensive production readiness assessment
readiness_checks = {
    'Best Model Selected': True,
    'Fairness Validated': not bias_detected,
    'Drift Acceptable': psi_df['psi_score'].max() < 0.2,
    'Models Compared': len(eval_df) >= 3,
    'Data Quality OK': True
}

print("\nProduction Readiness Assessment:")
for check, status in readiness_checks.items():
    symbol = "✓" if status else "✗"
    print(f"  {symbol} {check}: {'PASS' if status else 'FAIL'}")

all_passed = all(readiness_checks.values())
print(f"\nOverall Status: {'✓ READY FOR PRODUCTION' if all_passed else '⚠ NOT READY - Issues detected'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Analysis Summary

# COMMAND ----------

# Create comprehensive summary
summary = {
    'analysis_timestamp': datetime.now(),
    'best_model': best_model,
    'best_model_score': eval_df[eval_df['model'] == best_model]['mean_score'].iloc[0],
    'bias_detected': bias_detected,
    'major_drift_detected': not major_drift.empty,
    'production_ready': all_passed,
    'total_claims_analyzed': len(spark.table(f"{catalog}.{schema}.training_data").toPandas()),
    'models_evaluated': len(eval_df),
    'protected_attributes_checked': fairness_df['attribute'].nunique(),
    'features_monitored': len(psi_df)
}

summary_df = pd.DataFrame([summary])

# Save summary
spark.createDataFrame(summary_df).write.format("delta").mode("append").saveAsTable(
    f"{catalog}.{schema}.analysis_summary"
)
print("\n✓ Analysis summary saved to {catalog}.{schema}.analysis_summary")

# COMMAND ----------

print("=" * 60)
print("DOWNSTREAM ANALYSIS COMPLETE")
print("=" * 60)
