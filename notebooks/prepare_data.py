# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preparation
# MAGIC Loads and prepares data for training

# COMMAND ----------

print("=" * 60)
print("DATA PREPARATION")
print("=" * 60)

# Use workspace catalog (Unity Catalog)
catalog = "workspace"
schema = "default"

print(f"Using catalog: {catalog}")
print(f"Using schema: {schema}")

# COMMAND ----------

# Create sample training data
print("Creating sample training data...")
from pyspark.sql import Row

sample_data = [
    Row(
        claim_id=i, 
        claim_amount=float(1000 + i*10), 
        patient_age=30 + (i % 40),
        claim_type="Medical" if i % 2 == 0 else "Surgical",
        provider_specialty="Cardiology" if i % 3 == 0 else "Orthopedics"
    )
    for i in range(1000)
]

df = spark.createDataFrame(sample_data)
print(f"✓ Created {df.count()} sample rows")

# COMMAND ----------

# Save to Delta Lake
output_table = f"{catalog}.{schema}.training_data"
print(f"Saving to: {output_table}")
df.write.format("delta").mode("overwrite").saveAsTable(output_table)
print(f"✓ Saved to Delta table")

# COMMAND ----------

# Verify
result = spark.table(output_table)
print(f"\n✓ Verification: {result.count()} rows in training_data")
print("\nSample data:")
result.show(5)

# COMMAND ----------

# Store catalog info for downstream tasks
dbutils.jobs.taskValues.set(key="catalog", value=catalog)
dbutils.jobs.taskValues.set(key="schema", value=schema)
print(f"\n✓ Task values set: catalog={catalog}, schema={schema}")

# COMMAND ----------

print("=" * 60)
print("DATA PREPARATION COMPLETE")
print(f"Table: {output_table}")
print("=" * 60)
