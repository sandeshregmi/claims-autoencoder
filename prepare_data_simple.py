# Databricks notebook source
print("Starting data preparation...")

# Use workspace catalog with default schema
catalog = "workspace"
schema = "default"

print(f"Using catalog: {catalog}")
print(f"Using schema: {schema}")

# Create sample data
data = [(i, i*100, f"claim_{i}") for i in range(100)]
df = spark.createDataFrame(data, ["id", "value", "claim_id"])

# Save table
table_name = f"{catalog}.{schema}.test_data"
df.write.format("delta").mode("overwrite").saveAsTable(table_name)

print(f"✓ Created table: {table_name}")
print(f"✓ Row count: {df.count()}")

# Pass to next task
dbutils.jobs.taskValues.set(key="catalog", value=catalog)
dbutils.jobs.taskValues.set(key="schema", value=schema)

print("✓ Data preparation complete!")
