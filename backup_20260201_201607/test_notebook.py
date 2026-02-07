# Databricks notebook source
print("Hello from Databricks!")
print("Job is running successfully!")

# Test Spark
spark.sql("SELECT 'Success!' as status").show()
