# Databricks notebook source
# MAGIC %md
# MAGIC # Train CatBoost Model
# MAGIC Trains CatBoost autoencoder for fraud detection

# COMMAND ----------

print("=" * 60)
print("TRAINING CATBOOST MODEL")
print("=" * 60)

import pandas as pd
import numpy as np
from datetime import datetime

# COMMAND ----------

# Get catalog from upstream task or use default
try:
    catalog = dbutils.jobs.taskValues.get(taskKey="prepare_data", key="catalog")
    schema = dbutils.jobs.taskValues.get(taskKey="prepare_data", key="schema")
    print(f"✓ Using catalog from task: {catalog}.{schema}")
except:
    catalog = "hive_metastore"
    schema = "claims"
    print(f"⚠ Using default: {catalog}.{schema}")

# COMMAND ----------

# Load data
table_name = f"{catalog}.{schema}.training_data"
print(f"Loading data from: {table_name}")
data = spark.table(table_name).toPandas()
print(f"✓ Loaded {len(data)} rows")

# COMMAND ----------

# Simulate training (replace with your actual ClaimsTreeAutoencoder code)
print("Training CatBoost autoencoder...")
import time
time.sleep(5)  # Simulate training time

# Generate fake fraud scores for testing
fraud_scores = np.random.gamma(2, 100, len(data))
mean_score = fraud_scores.mean()
p95_score = np.percentile(fraud_scores, 95)

print(f"✓ Training complete!")
print(f"  Mean fraud score: {mean_score:.2f}")
print(f"  P95 fraud score: {p95_score:.2f}")

# COMMAND ----------

# Save results
results_df = data.copy()
results_df['fraud_score'] = fraud_scores
results_df['model_type'] = 'catboost'
results_df['trained_at'] = datetime.now()

output_table = f"{catalog}.{schema}.fraud_scores_catboost"
spark_df = spark.createDataFrame(results_df)
spark_df.write.format("delta").mode("overwrite").saveAsTable(output_table)
print(f"✓ Saved results to {output_table}")

# COMMAND ----------

# Log metrics (simulate MLflow)
print("\nMetrics:")
print(f"  model_type: catboost")
print(f"  mean_score: {mean_score:.2f}")
print(f"  p95_score: {p95_score:.2f}")
print(f"  n_samples: {len(data)}")

# COMMAND ----------

print("=" * 60)
print("CATBOOST TRAINING COMPLETE")
print("=" * 60)
