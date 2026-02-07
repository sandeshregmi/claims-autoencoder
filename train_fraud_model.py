# Databricks notebook source
import pandas as pd
from src.tree_models import ClaimsTreeAutoencoder

# Load data from Delta Lake
data = spark.table("dev_fraud_detection.claims.training_data").toPandas()

# Train model
model = ClaimsTreeAutoencoder(model_type='catboost')
fraud_scores, errors = model.fit(data, cat_features, num_features)

print(f"Training complete! Mean fraud score: {fraud_scores.mean()}")
