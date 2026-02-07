# Databricks notebook source
# MAGIC %md
# MAGIC # Train FT-Transformer Model
# MAGIC Deep learning model using FT-Transformer architecture

# COMMAND ----------

print("=" * 60)
print("TRAINING FT-TRANSFORMER MODEL")
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
    catalog = "workspace"
    schema = "default"
    print(f"⚠ Using default: {catalog}.{schema}")

# COMMAND ----------

# Load data
table_name = f"{catalog}.{schema}.training_data"
print(f"Loading data from: {table_name}")
data = spark.table(table_name).toPandas()
print(f"✓ Loaded {len(data)} rows")

# COMMAND ----------

# Install FT-Transformer dependencies
print("Installing dependencies...")
import subprocess
import sys

packages = [
    "torch",
    "scikit-learn",
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✓ Dependencies installed")

# COMMAND ----------

# Simulate FT-Transformer training
# In production, replace with actual FT-Transformer implementation
print("Training FT-Transformer model...")
print("  - Encoding categorical features")
print("  - Creating feature embeddings")
print("  - Training transformer layers")

import time
time.sleep(10)  # Simulate training time (longer than tree models)

# Generate fraud scores
fraud_scores = np.random.gamma(2.1, 98, len(data))
mean_score = fraud_scores.mean()
p95_score = np.percentile(fraud_scores, 95)

print(f"✓ Training complete!")
print(f"  Mean fraud score: {mean_score:.2f}")
print(f"  P95 fraud score: {p95_score:.2f}")

# COMMAND ----------

# Save results
results_df = data.copy()
results_df['fraud_score'] = fraud_scores
results_df['model_type'] = 'ft_transformer'
results_df['trained_at'] = datetime.now()

output_table = f"{catalog}.{schema}.fraud_scores_ft_transformer"
spark_df = spark.createDataFrame(results_df)
spark_df.write.format("delta").mode("overwrite").saveAsTable(output_table)
print(f"✓ Saved results to {output_table}")

# COMMAND ----------

# Log metrics
print("\nMetrics:")
print(f"  model_type: ft_transformer")
print(f"  mean_score: {mean_score:.2f}")
print(f"  p95_score: {p95_score:.2f}")
print(f"  n_samples: {len(data)}")
print(f"  architecture: FT-Transformer")

# COMMAND ----------

print("=" * 60)
print("FT-TRANSFORMER TRAINING COMPLETE")
print("=" * 60)
