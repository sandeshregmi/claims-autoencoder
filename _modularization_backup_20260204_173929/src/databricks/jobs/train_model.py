"""
Databricks Training Job Entry Point
Orchestrates model training with MLflow tracking and coordination hooks
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from pyspark.sql import SparkSession
    import mlflow
    import numpy as np
    
    from src.tree_models import ClaimsTreeAutoencoder
    from src.coordination_hooks import coordination_hooks
    
    DATABRICKS_MODE = True
except ImportError as e:
    print(f"Warning: {e}")
    print("Running in local mode (some features disabled)")
    DATABRICKS_MODE = False


def main():
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--model-type', choices=['catboost', 'xgboost'], required=True)
    parser.add_argument('--catalog', required=True, help='Unity Catalog name')
    parser.add_argument('--schema', required=True, help='Schema name')
    parser.add_argument('--experiment-name', required=True, help='MLflow experiment path')
    parser.add_argument('--run-name', required=True, help='MLflow run name')
    args = parser.parse_args()
    
    if not DATABRICKS_MODE:
        print("ERROR: This script requires Databricks environment")
        sys.exit(1)
    
    # Initialize Spark
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName(f"Claims Fraud Training - {args.model_type}") \
        .getOrCreate()
    
    # Pre-training coordination hook
    print(f"🔔 Starting {args.model_type} training...")
    coordination_hooks.pre_model_training(args.model_type)
    
    try:
        # Load data from Delta Lake
        feature_table = f"{args.catalog}.{args.schema}.features"
        print(f"Loading features from: {feature_table}")
        
        data = spark.table(feature_table).toPandas()
        print(f"✅ Loaded {len(data)} rows")
        
        # Define features
        cat_features = [
            'claim_type',
            'provider_specialty',
            'diagnosis_code',
            'procedure_code',
            'patient_gender',
            'geographic_region'
        ]
        
        num_features = [
            'claim_amount',
            'patient_age',
            'provider_experience_years',
            'days_since_last_claim',
            'num_previous_claims',
            'average_claim_amount',
            'claim_duration_days'
        ]
        
        print(f"Features: {len(cat_features)} categorical, {len(num_features)} numerical")
        
        # Set MLflow experiment
        mlflow.set_experiment(args.experiment_name)
        print(f"MLflow experiment: {args.experiment_name}")
        
        # Train model with MLflow logging
        print(f"Training {args.model_type} autoencoder...")
        model = ClaimsTreeAutoencoder(model_type=args.model_type)
        
        with mlflow.start_run(run_name=args.run_name) as run:
            # Log parameters
            mlflow.log_param("model_type", args.model_type)
            mlflow.log_param("n_samples", len(data))
            mlflow.log_param("n_features", len(cat_features) + len(num_features))
            mlflow.log_param("catalog", args.catalog)
            mlflow.log_param("schema", args.schema)
            
            # Train
            model.fit(data, cat_features, num_features, verbose=False)
            
            # Compute fraud scores
            fraud_scores, per_feature_errors = model.compute_fraud_scores(data)
            
            # Log metrics
            mlflow.log_metric("mean_fraud_score", float(fraud_scores.mean()))
            mlflow.log_metric("median_fraud_score", float(np.median(fraud_scores)))
            mlflow.log_metric("p95_fraud_score", float(np.percentile(fraud_scores, 95)))
            mlflow.log_metric("p99_fraud_score", float(np.percentile(fraud_scores, 99)))
            mlflow.log_metric("max_fraud_score", float(fraud_scores.max()))
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"{args.catalog}.{args.schema}.{args.model_type}_autoencoder"
            )
            
            print(f"✅ Model logged to MLflow")
            print(f"   Run ID: {run.info.run_id}")
            print(f"   Mean fraud score: {fraud_scores.mean():.2f}")
            print(f"   P95 fraud score: {np.percentile(fraud_scores, 95):.2f}")
            
            # Save fraud scores to Delta
            results_df = data.copy()
            results_df['fraud_score'] = fraud_scores
            results_df['run_id'] = run.info.run_id
            results_df['model_type'] = args.model_type
            
            output_table = f"{args.catalog}.{args.schema}.fraud_scores_{args.model_type}"
            
            spark.createDataFrame(results_df).write \
                .format("delta") \
                .mode("overwrite") \
                .option("mergeSchema", "true") \
                .saveAsTable(output_table)
            
            print(f"✅ Fraud scores saved to: {output_table}")
            
            # Post-training coordination hook
            coordination_hooks.post_model_training(args.model_type, {
                "mean_score": float(fraud_scores.mean()),
                "p95_score": float(np.percentile(fraud_scores, 95)),
                "run_id": run.info.run_id
            })
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        spark.stop()
    
    print(f"✅ Training complete for {args.model_type}")


if __name__ == "__main__":
    main()
