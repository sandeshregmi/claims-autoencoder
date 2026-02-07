# Databricks notebook source
# MAGIC %md
# MAGIC # Fairness Monitoring
# MAGIC Continuous monitoring for model fairness

# COMMAND ----------

print("=" * 60)
print("FAIRNESS MONITORING CHECK")
print("=" * 60)

import pandas as pd

# COMMAND ----------

catalog = "dev_fraud_detection"
schema = "claims"

# Check latest fairness results
try:
    fairness_df = spark.table(f"{catalog}.{schema}.fairness_results").toPandas()
    
    print(f"✓ Latest fairness check:")
    print(fairness_df[['attribute', 'group', 'disparate_impact', 'is_fair']])
    
    bias_detected = not fairness_df['is_fair'].all()
    
    if bias_detected:
        print(f"\n⚠ ALERT: Bias detected!")
        biased = fairness_df[~fairness_df['is_fair']]
        print(biased)
        print(f"\n  → Action: Review model and data for bias")
    else:
        print(f"\n✓ No bias detected - Model is fair across all groups")
        
except Exception as e:
    print(f"⚠ Could not load fairness results: {e}")
    print("  Schedule fairness analysis to run")

# COMMAND ----------

print("=" * 60)
print("FAIRNESS CHECK COMPLETE")
print("=" * 60)
