# Databricks notebook source
# MAGIC %md
# MAGIC # Drift Monitoring
# MAGIC Continuous monitoring for data drift

# COMMAND ----------

print("=" * 60)
print("DRIFT MONITORING CHECK")
print("=" * 60)

import pandas as pd
import numpy as np
from datetime import datetime

# COMMAND ----------

catalog = "dev_fraud_detection"
schema = "claims"

# Check latest PSI results
try:
    psi_df = spark.table(f"{catalog}.{schema}.psi_results").toPandas()
    max_psi = psi_df['psi_score'].max()
    
    print(f"✓ Latest PSI check:")
    print(f"  Max PSI score: {max_psi:.3f}")
    
    if max_psi >= 0.2:
        print(f"\n⚠ ALERT: Major drift detected (PSI >= 0.2)")
        print(f"  → Action: Consider retraining model")
    elif max_psi >= 0.1:
        print(f"\n⚠ WARNING: Medium drift detected (PSI >= 0.1)")
        print(f"  → Action: Monitor closely")
    else:
        print(f"\n✓ No significant drift detected")
        
except Exception as e:
    print(f"⚠ Could not load PSI results: {e}")
    print("  Running initial drift check...")

# COMMAND ----------

print("=" * 60)
print("DRIFT CHECK COMPLETE")
print("=" * 60)
