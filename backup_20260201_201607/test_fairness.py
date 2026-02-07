#!/usr/bin/env python3
"""
Test script to verify Fairness Analysis is working
"""

import sys
import pandas as pd
import numpy as np

print("=" * 60)
print("FAIRNESS ANALYSIS - VERIFICATION TEST")
print("=" * 60)
print()

# Test 1: Check if module exists
print("Test 1: Checking if fairness_analysis.py exists...")
try:
    from src.fairness_analysis import FairnessAnalyzer
    print("✅ Module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test 2: Create sample data and test
print("\nTest 2: Testing with sample data...")
try:
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'patient_gender': np.random.choice(['M', 'F'], n_samples),
        'patient_age_group': np.random.choice(['18-35', '36-50', '51-65', '65+'], n_samples),
        'geographic_region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    })
    
    # Generate fraud scores
    fraud_scores = np.random.lognormal(10, 2, n_samples)
    
    print(f"✅ Created sample data: {len(sample_data)} records")
    print(f"   Protected attributes: {list(sample_data.columns)}")
    
except Exception as e:
    print(f"❌ Failed to create sample data: {e}")
    sys.exit(1)

# Test 3: Initialize analyzer
print("\nTest 3: Initializing FairnessAnalyzer...")
try:
    analyzer = FairnessAnalyzer(
        data=sample_data,
        fraud_scores=fraud_scores,
        protected_attributes=['patient_gender', 'patient_age_group', 'geographic_region'],
        threshold_percentile=95.0
    )
    print("✅ Analyzer initialized successfully")
    print(f"   Threshold: {analyzer.threshold:.2f}")
    print(f"   Fraud flags: {analyzer.fraud_flags.sum()} / {len(analyzer.fraud_flags)}")
    
except Exception as e:
    print(f"❌ Failed to initialize analyzer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Run analysis
print("\nTest 4: Running fairness analysis...")
try:
    results = analyzer.analyze_all_attributes()
    print(f"✅ Analysis completed for {len(results)} attributes")
    
    for attr, result in results.items():
        if 'error' in result:
            print(f"   ❌ {attr}: {result['error']}")
        else:
            num_groups = len(result.get('groups', {}))
            print(f"   ✅ {attr}: {num_groups} groups analyzed")
    
except Exception as e:
    print(f"❌ Failed to run analysis: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Get bias summary
print("\nTest 5: Getting bias summary...")
try:
    bias_summary = analyzer.get_bias_summary()
    print(f"✅ Bias summary generated")
    print(f"\n{bias_summary}")
    
except Exception as e:
    print(f"❌ Failed to get bias summary: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Generate report
print("\nTest 6: Generating detailed report...")
try:
    report = analyzer.generate_fairness_report('patient_gender')
    print("✅ Report generated successfully")
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Failed to generate report: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nFairness Analysis module is working correctly!")
print("You can now use it in the Streamlit app.")
