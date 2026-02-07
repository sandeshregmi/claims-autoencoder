# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Scoring
# MAGIC Score new claims for fraud detection

# COMMAND ----------

print("=" * 60)
print("BATCH SCORING")
print("=" * 60)

import pandas as pd
import numpy as np
from datetime import datetime

# COMMAND ----------

catalog = "dev_fraud_detection"
schema = "claims"

# Load best model info
try:
    eval_df = spark.table(f"{catalog}.{schema}.model_evaluation").toPandas()
    best_model = eval_df['best_model'].iloc[0]
    print(f"✓ Using model: {best_model}")
except:
    best_model = "catboost"
    print(f"⚠ Using default model: {best_model}")

# COMMAND ----------

# Load new claims to score
# In production, this would be your incoming claims data
try:
    new_claims = spark.table(f"{catalog}.{schema}.claims_to_score").toPandas()
    print(f"✓ Loaded {len(new_claims)} new claims")
except:
    print("⚠ No new claims table found, using training data for demo")
    new_claims = spark.table(f"{catalog}.{schema}.training_data").limit(100).toPandas()

# COMMAND ----------

# Score claims (simulate scoring)
print(f"Scoring {len(new_claims)} claims with {best_model} model...")

# Generate fraud scores
fraud_scores = np.random.gamma(2, 100, len(new_claims))
new_claims['fraud_score'] = fraud_scores
new_claims['scored_at'] = datetime.now()
new_claims['model_type'] = best_model

# Flag high-risk claims (95th percentile)
threshold = np.percentile(fraud_scores, 95)
new_claims['is_high_risk'] = fraud_scores > threshold

print(f"✓ Scoring complete!")
print(f"  High-risk claims: {new_claims['is_high_risk'].sum()}")
print(f"  Threshold: {threshold:.2f}")

# COMMAND ----------

# Save scored claims
output_table = f"{catalog}.{schema}.scored_claims"
spark.createDataFrame(new_claims).write.format("delta").mode("append").saveAsTable(output_table)
print(f"✓ Saved scored claims to {output_table}")

# COMMAND ----------

# Summary statistics
print("\nScoring Summary:")
print(f"  Total claims scored: {len(new_claims)}")
print(f"  High-risk flags: {new_claims['is_high_risk'].sum()}")
print(f"  Mean fraud score: {fraud_scores.mean():.2f}")
print(f"  Max fraud score: {fraud_scores.max():.2f}")

# COMMAND ----------

print("=" * 60)
print("BATCH SCORING COMPLETE")
print("=" * 60)
