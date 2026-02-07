# Databricks notebook source
# MAGIC %md
# MAGIC # Model Evaluation
# MAGIC Compare all three models: CatBoost, XGBoost, and FT-Transformer

# COMMAND ----------

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

import pandas as pd
import numpy as np

# COMMAND ----------

# Get catalog
try:
    catalog = dbutils.jobs.taskValues.get(taskKey="prepare_data", key="catalog")
    schema = dbutils.jobs.taskValues.get(taskKey="prepare_data", key="schema")
except:
    catalog = "workspace"
    schema = "default"

print(f"Using: {catalog}.{schema}")

# COMMAND ----------

# Load all model results
catboost_df = spark.table(f"{catalog}.{schema}.fraud_scores_catboost").toPandas()
xgboost_df = spark.table(f"{catalog}.{schema}.fraud_scores_xgboost").toPandas()
ft_transformer_df = spark.table(f"{catalog}.{schema}.fraud_scores_ft_transformer").toPandas()

print(f"✓ Loaded CatBoost results: {len(catboost_df)} rows")
print(f"✓ Loaded XGBoost results: {len(xgboost_df)} rows")
print(f"✓ Loaded FT-Transformer results: {len(ft_transformer_df)} rows")

# COMMAND ----------

# Compare metrics
models = {
    'catboost': catboost_df['fraud_score'],
    'xgboost': xgboost_df['fraud_score'],
    'ft_transformer': ft_transformer_df['fraud_score']
}

comparison = []
for name, scores in models.items():
    comparison.append({
        'model': name,
        'mean_score': scores.mean(),
        'median_score': scores.median(),
        'std_score': scores.std(),
        'p95_score': np.percentile(scores, 95),
        'p99_score': np.percentile(scores, 99)
    })

comparison_df = pd.DataFrame(comparison)
print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

# COMMAND ----------

# Select best model (highest mean score)
best_idx = comparison_df['mean_score'].idxmax()
best_model = comparison_df.loc[best_idx, 'model']
best_score = comparison_df.loc[best_idx, 'mean_score']

print(f"\n✓ Best model: {best_model} (mean score: {best_score:.2f})")

# COMMAND ----------

# Save evaluation results
eval_results = comparison_df.copy()
eval_results['best_model'] = best_model
eval_results['evaluation_timestamp'] = pd.Timestamp.now()

spark.createDataFrame(eval_results).write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog}.{schema}.model_evaluation"
)
print(f"✓ Saved evaluation to {catalog}.{schema}.model_evaluation")

# Set for downstream tasks
dbutils.jobs.taskValues.set(key="best_model", value=best_model)

# COMMAND ----------

print("=" * 60)
print("EVALUATION COMPLETE")
print(f"Best Model: {best_model}")
print("=" * 60)
