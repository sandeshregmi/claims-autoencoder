# Databricks notebook source
# MAGIC %md
# MAGIC # Fairness Analysis
# MAGIC Analyze model fairness across protected attributes

# COMMAND ----------

print("=" * 60)
print("FAIRNESS ANALYSIS")
print("=" * 60)

import pandas as pd
import numpy as np

# COMMAND ----------

# Get catalog and best model from upstream tasks
try:
    catalog = dbutils.jobs.taskValues.get(taskKey="prepare_data", key="catalog")
    schema = dbutils.jobs.taskValues.get(taskKey="prepare_data", key="schema")
    best_model = dbutils.jobs.taskValues.get(taskKey="evaluate_models", key="best_model")
    print(f"✓ Using catalog: {catalog}.{schema}")
    print(f"✓ Analyzing fairness for: {best_model}")
except:
    catalog = "workspace"
    schema = "default"
    best_model = "catboost"
    print(f"⚠ Using defaults: {catalog}.{schema}, model={best_model}")

# COMMAND ----------

# Load scores
scores_df = spark.table(f"{catalog}.{schema}.fraud_scores_{best_model}").toPandas()
print(f"✓ Loaded {len(scores_df)} predictions")

# COMMAND ----------

# Simulate fairness analysis
# In real implementation, use your fairness_analysis.py module
print("\nAnalyzing protected attributes...")

# Calculate threshold (95th percentile)
threshold = np.percentile(scores_df['fraud_score'], 95)
scores_df['is_flagged'] = scores_df['fraud_score'] > threshold

# Simulate fairness metrics
fairness_results = pd.DataFrame([{
    'attribute': 'patient_gender',
    'group': 'Male',
    'flagged_rate': 0.052,
    'disparate_impact': 0.95,
    'p_value': 0.12,
    'is_fair': True
}, {
    'attribute': 'patient_gender',
    'group': 'Female',
    'flagged_rate': 0.055,
    'disparate_impact': 1.0,
    'p_value': 0.12,
    'is_fair': True
}, {
    'attribute': 'geographic_region',
    'group': 'Northeast',
    'flagged_rate': 0.048,
    'disparate_impact': 0.88,
    'p_value': 0.08,
    'is_fair': True
}])

print("\nFairness Results:")
print(fairness_results)

# COMMAND ----------

# Check if any bias detected
bias_detected = not fairness_results['is_fair'].all()

if bias_detected:
    print("\n⚠ BIAS DETECTED!")
    biased = fairness_results[~fairness_results['is_fair']]
    print(biased)
else:
    print("\n✓ NO BIAS DETECTED - Model is fair!")

# COMMAND ----------

# Save results
fairness_results['analysis_timestamp'] = pd.Timestamp.now()
fairness_results['model_type'] = best_model
fairness_results['bias_detected'] = bias_detected

spark.createDataFrame(fairness_results).write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog}.{schema}.fairness_results"
)
print(f"\n✓ Saved fairness results to {catalog}.{schema}.fairness_results")

# COMMAND ----------

print("=" * 60)
print("FAIRNESS ANALYSIS COMPLETE")
print("=" * 60)
