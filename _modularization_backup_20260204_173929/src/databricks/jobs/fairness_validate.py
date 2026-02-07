"""
Databricks Fairness Validation Job
Validates model fairness across protected attributes with automated alerts
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from pyspark.sql import SparkSession
    import pandas as pd
    import numpy as np
    
    from src.fairness_analysis import FairnessAnalyzer
    from src.coordination_hooks import coordination_hooks
    
    DATABRICKS_MODE = True
except ImportError as e:
    print(f"Warning: {e}")
    DATABRICKS_MODE = False


def send_alert(message: str, severity: str = "warning"):
    """Send alert (placeholder - integrate with your alerting system)"""
    print(f"🚨 ALERT [{severity.upper()}]: {message}")
    # TODO: Integrate with Slack, email, or other alerting system
    # Example:
    # requests.post(webhook_url, json={"text": message})


def main():
    parser = argparse.ArgumentParser(description='Validate model fairness')
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--protected-attributes', required=True, 
                       help='Comma-separated list of protected attributes')
    parser.add_argument('--threshold-percentile', type=float, default=95.0)
    parser.add_argument('--output-table', required=True)
    parser.add_argument('--fail-on-bias', action='store_true', 
                       help='Fail job if bias detected')
    args = parser.parse_args()
    
    if not DATABRICKS_MODE:
        print("ERROR: This script requires Databricks environment")
        sys.exit(1)
    
    # Parse protected attributes
    protected_attrs = [attr.strip() for attr in args.protected_attributes.split(',')]
    
    # Initialize Spark
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("Claims Fraud - Fairness Validation") \
        .getOrCreate()
    
    # Pre-fairness coordination hook
    print(f"🔔 Starting fairness analysis for: {protected_attrs}")
    coordination_hooks.pre_fairness_analysis(protected_attrs)
    
    try:
        # Load data
        feature_table = f"{args.catalog}.{args.schema}.features"
        print(f"Loading features from: {feature_table}")
        data = spark.table(feature_table).toPandas()
        
        # Load fraud scores (use CatBoost by default)
        scores_table = f"{args.catalog}.{args.schema}.fraud_scores_catboost"
        print(f"Loading fraud scores from: {scores_table}")
        
        try:
            fraud_scores_df = spark.table(scores_table).toPandas()
            fraud_scores = fraud_scores_df['fraud_score'].values
        except Exception:
            print(f"Warning: {scores_table} not found, trying XGBoost scores")
            scores_table = f"{args.catalog}.{args.schema}.fraud_scores_xgboost"
            fraud_scores_df = spark.table(scores_table).toPandas()
            fraud_scores = fraud_scores_df['fraud_score'].values
        
        print(f"✅ Loaded {len(data)} rows, {len(fraud_scores)} fraud scores")
        
        # Create age groups if needed
        if 'patient_age' in data.columns and 'patient_age_group' not in data.columns:
            data['patient_age_group'] = pd.cut(
                data['patient_age'],
                bins=[0, 30, 45, 60, 100],
                labels=['<30', '30-45', '45-60', '60+']
            )
            print("✅ Created patient_age_group")
        
        # Run fairness analysis
        print(f"Running fairness analysis (threshold={args.threshold_percentile}%)...")
        analyzer = FairnessAnalyzer(
            data=data,
            fraud_scores=fraud_scores,
            protected_attributes=protected_attrs,
            threshold_percentile=args.threshold_percentile
        )
        
        results = analyzer.analyze_all_attributes()
        
        # Post-fairness coordination hook
        coordination_hooks.post_fairness_analysis(results)
        
        # Analyze results
        biased_attributes = []
        for attr, result in results.items():
            if 'error' in result:
                print(f"⚠️ Error analyzing {attr}: {result['error']}")
                continue
            
            if 'overall_metrics' in result:
                is_fair = result['overall_metrics'].get('is_fair', True)
                min_di = result['overall_metrics'].get('min_disparate_impact_ratio', 1.0)
                max_di = result['overall_metrics'].get('max_disparate_impact_ratio', 1.0)
                
                if not is_fair:
                    biased_attributes.append(attr)
                    print(f"❌ BIAS DETECTED in {attr}")
                    print(f"   DI ratios: {min_di:.3f} - {max_di:.3f}")
                    
                    # Send alert
                    send_alert(
                        f"Bias detected in {attr}. DI ratios: {min_di:.3f} - {max_di:.3f}",
                        severity="critical"
                    )
                else:
                    print(f"✅ {attr} is fair (DI ratios: {min_di:.3f} - {max_di:.3f})")
        
        # Save results to Delta
        bias_summary = analyzer.get_bias_summary()
        
        if not bias_summary.empty:
            # Add timestamp
            bias_summary['analysis_timestamp'] = pd.Timestamp.now()
            bias_summary['threshold_percentile'] = args.threshold_percentile
            
            spark.createDataFrame(bias_summary).write \
                .format("delta") \
                .mode("overwrite") \
                .option("mergeSchema", "true") \
                .saveAsTable(args.output_table)
            
            print(f"✅ Fairness results saved to: {args.output_table}")
        
        # Summary
        total_attrs = len([r for r in results.values() if 'error' not in r])
        biased_count = len(biased_attributes)
        fair_count = total_attrs - biased_count
        
        print("\n" + "="*60)
        print("FAIRNESS ANALYSIS SUMMARY")
        print("="*60)
        print(f"Attributes analyzed: {total_attrs}")
        print(f"Fair attributes: {fair_count}")
        print(f"Biased attributes: {biased_count}")
        
        if biased_attributes:
            print(f"\nBiased attributes: {', '.join(biased_attributes)}")
            print("\n⚠️ BIAS DETECTED - REVIEW REQUIRED")
            
            # Send summary alert
            send_alert(
                f"Fairness validation complete: {biased_count}/{total_attrs} attributes show bias. "
                f"Biased: {', '.join(biased_attributes)}",
                severity="critical"
            )
            
            if args.fail_on_bias:
                print("\n❌ FAILING JOB DUE TO BIAS DETECTION")
                sys.exit(1)
        else:
            print("\n✅ NO BIAS DETECTED - MODEL IS FAIR")
            send_alert(
                f"Fairness validation passed: All {total_attrs} attributes are fair",
                severity="info"
            )
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Fairness validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        send_alert(
            f"Fairness validation job failed: {str(e)}",
            severity="critical"
        )
        
        sys.exit(1)
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
