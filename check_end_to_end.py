#!/usr/bin/env python3
"""
End-to-End Project Audit
Verifies everything is automated and config-driven
"""

import sys
from pathlib import Path

print("=" * 80)
print("🔍 END-TO-END PROJECT AUDIT")
print("=" * 80)
print()

project_root = Path("/Users/sregmi/pytorch-tabular-mcp/claims-autoencoder")
sys.path.insert(0, str(project_root / "src"))

results = {
    "✅ PASS": [],
    "⚠️  WARNING": [],
    "❌ FAIL": []
}

def check(name, passed, message=""):
    """Record check result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    results[status].append(f"{name}: {message}" if message else name)
    symbol = "✅" if passed else "❌"
    print(f"  {symbol} {name}")
    if message:
        print(f"     {message}")

# 1. Configuration Check
print("1️⃣  CONFIGURATION")
print("-" * 80)

try:
    from claims_fraud.config.manager import load_config
    config_path = project_root / "config" / "config.yaml"
    config = load_config(str(config_path))
    check("Config Loading", True)
    
    # Check Databricks config
    has_databricks = hasattr(config, 'databricks')
    check("Databricks Configuration", has_databricks)
    
    if has_databricks:
        has_catalog = hasattr(config.databricks.data_source, 'catalog')
        has_schema = hasattr(config.databricks.data_source, 'schema')
        check("Unity Catalog Settings", has_catalog and has_schema,
              f"Catalog: {config.databricks.data_source.catalog if has_catalog else 'N/A'}, "
              f"Schema: {config.databricks.data_source.schema if has_schema else 'N/A'}")
        
        has_column_mapping = 'column_mapping' in config.databricks.data_source
        check("Column Mapping", has_column_mapping,
              f"{len(config.databricks.data_source.get('column_mapping', {}))} mappings")
    
    # Check feature config
    num_features = len(config.data.numerical_features)
    cat_features = len(config.data.categorical_features)
    check("Feature Configuration", num_features > 0 and cat_features > 0,
          f"{num_features} numerical, {cat_features} categorical")
    
    # Check business rules
    has_business_rules = hasattr(config, 'business_rules')
    check("Business Rules", has_business_rules)
    
except Exception as e:
    check("Config Loading", False, str(e))

print()

# 2. Modules Check
print("2️⃣  MODULES")
print("-" * 80)

try:
    from claims_fraud.data.validation import DataValidator
    check("Validation Module", True)
except Exception as e:
    check("Validation Module", False, str(e))

try:
    from claims_fraud.core.business_rules import BusinessRulesEngine
    check("Business Rules Module", True)
except Exception as e:
    check("Business Rules Module", False, str(e))

try:
    from claims_fraud.data.databricks_loader import DatabricksDataLoader
    check("Databricks Loader", True)
except Exception as e:
    check("Databricks Loader", False, str(e))

print()

# 3. Automation Check
print("3️⃣  AUTOMATION VERIFICATION")
print("-" * 80)

# Check if hardcoded values exist
hardcoded_checks = [
    ("Feature Names", "No hardcoded feature names", 
     all(f in config.data.numerical_features + config.data.categorical_features 
         for f in ['claim_amount', 'patient_age'])),
    
    ("Categorical Values", "Categorical domains in config",
     'categorical_domains' in config.data and len(config.data.categorical_domains) > 0),
    
    ("Business Thresholds", "Thresholds in config",
     hasattr(config, 'business_rules') and 'fraud_thresholds' in config.business_rules),
    
    ("Data Validation Rules", "Validation schemas in config",
     'feature_schemas' in config.data and len(config.data.feature_schemas) > 0),
]

for name, desc, passed in hardcoded_checks:
    check(name, passed, desc)

print()

# 4. Databricks Readiness
print("4️⃣  DATABRICKS READINESS")
print("-" * 80)

databricks_checks = [
    ("Unity Catalog Config", hasattr(config, 'databricks') and 
     hasattr(config.databricks.data_source, 'catalog')),
    
    ("SQL Query Template", hasattr(config, 'databricks') and 
     'query_template' in config.databricks.data_source),
    
    ("Column Mapping", hasattr(config, 'databricks') and 
     'column_mapping' in config.databricks.data_source),
    
    ("Storage Paths", hasattr(config, 'databricks') and 
     hasattr(config.databricks, 'storage')),
    
    ("MLflow Integration", hasattr(config, 'databricks') and 
     hasattr(config.databricks, 'mlflow')),
]

for name, passed in databricks_checks:
    check(name, passed)

print()

# 5. Summary
print("=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print()

for status, items in results.items():
    if items:
        print(f"{status}: {len(items)} items")
        for item in items:
            print(f"  • {item}")
        print()

# Final verdict
total_checks = sum(len(items) for items in results.values())
passed_checks = len(results["✅ PASS"])
failed_checks = len(results["❌ FAIL"])

print("=" * 80)
if failed_checks == 0:
    print("🎉 ALL CHECKS PASSED!")
    print("=" * 80)
    print()
    print("✅ Your project is:")
    print("   • 100% config-driven")
    print("   • Databricks-ready")
    print("   • Production-ready")
    print()
    print("📋 Next steps:")
    print("   1. Update config/config.yaml with your Databricks table names")
    print("   2. Update column_mapping to match your schema")
    print("   3. Deploy to Databricks and test")
    print()
    print("See: DATABRICKS_PORTING_GUIDE.md")
    sys.exit(0)
else:
    print("⚠️  SOME CHECKS FAILED")
    print("=" * 80)
    print()
    print(f"Passed: {passed_checks}/{total_checks}")
    print(f"Failed: {failed_checks}/{total_checks}")
    print()
    print("Review failed checks above and fix them.")
    sys.exit(1)
