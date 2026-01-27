"""
Quick SHAP Force Plot Demo
Creates waterfall and force plots for top fraudulent claims
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    shap.initjs()  # Initialize JavaScript for interactive plots
except ImportError:
    print("ERROR: SHAP not installed!")
    print("Install with: pip install shap")
    exit(1)

from src.tree_models import ClaimsTreeAutoencoder
from src.config_manager import ConfigManager
from src.data_ingestion import DataIngestion

print("\n" + "="*80)
print("SHAP FORCE PLOT GENERATOR")
print("="*80)

# Load data
print("\n1. Loading data...")
config_manager = ConfigManager("config/example_config.yaml")
config = config_manager.get_config()

data_ingestion = DataIngestion(config)
train_df, val_df, test_df = data_ingestion.load_train_val_test()
print(f"   ✓ Loaded {len(val_df)} validation claims")

# Train model
print("\n2. Training CatBoost model...")
model = ClaimsTreeAutoencoder(model_type='catboost')
cat_features = config.data.categorical_features
num_features = config.data.numerical_features

model.fit(train_df, cat_features, num_features, verbose=False)
print("   ✓ Model trained")

# Compute fraud scores
print("\n3. Computing fraud scores...")
fraud_scores, _ = model.compute_fraud_scores(val_df)
print(f"   ✓ Scores computed (mean: {fraud_scores.mean():,.0f})")

# Get top fraudulent claim
top_idx = fraud_scores.argmax()
print(f"\n4. Most fraudulent claim: Index {top_idx}, Score: {fraud_scores[top_idx]:,.0f}")

# Prepare data for SHAP
target_feature = 'claim_amount'
predictor_features = [f for f in model.feature_names if f != target_feature]

X = val_df[predictor_features].copy()
cat_predictors = [f for f in predictor_features if f in cat_features]

# Convert categorical to numeric
for col in cat_predictors:
    if col in X.columns:
        X[col] = pd.Categorical(X[col]).codes

# Fill NaN
X = X.fillna(0)

# Get model for this feature
tree_model = model.models[target_feature]

print("\n5. Creating SHAP explainer...")
explainer = shap.TreeExplainer(tree_model)

# Get top 10 claims for analysis
top_10_indices = fraud_scores.argsort()[-10:][::-1]
X_top10 = X.iloc[top_10_indices]

print("   Computing SHAP values for top 10 claims...")
shap_values = explainer.shap_values(X_top10)

# Handle multi-output
if isinstance(shap_values, list):
    shap_values = shap_values[0]

print("   ✓ SHAP values computed")

# Create output directory
import os
os.makedirs("results/shap_plots", exist_ok=True)

print("\n6. Creating visualizations...")

# PLOT 1: Waterfall for most fraudulent claim
print("   Creating waterfall plot...")
plt.figure(figsize=(10, 8))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[0],
        data=X_top10.iloc[0].values,
        feature_names=predictor_features
    ),
    show=False
)
plt.title(f"SHAP Waterfall - Most Fraudulent Claim (Score: {fraud_scores[top_10_indices[0]]:,.0f})", 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/shap_plots/waterfall_top1.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/shap_plots/waterfall_top1.png")

# PLOT 2: Force plot for top claim
print("   Creating force plot...")
plt.figure(figsize=(20, 3))
shap.plots.force(
    explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[0],
    shap_values[0],
    X_top10.iloc[0],
    matplotlib=True,
    show=False,
    feature_names=predictor_features
)
plt.title(f"SHAP Force Plot - Most Fraudulent Claim", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/shap_plots/force_plot_top1.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/shap_plots/force_plot_top1.png")

# PLOT 3: Bar plot for top claim
print("   Creating bar plot...")
plt.figure(figsize=(10, 8))
shap.plots.bar(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[0],
        data=X_top10.iloc[0].values,
        feature_names=predictor_features
    ),
    show=False,
    max_display=15
)
plt.title(f"Feature Importance - Most Fraudulent Claim", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/shap_plots/bar_plot_top1.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/shap_plots/bar_plot_top1.png")

# PLOT 4: Summary plot for all top 10
print("   Creating summary plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_top10, show=False, max_display=15)
plt.title("SHAP Summary - Top 10 Fraudulent Claims", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/shap_plots/summary_top10.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/shap_plots/summary_top10.png")

# PLOT 5: Beeswarm plot
print("   Creating beeswarm plot...")
plt.figure(figsize=(12, 10))
shap.plots.beeswarm(
    shap.Explanation(
        values=shap_values,
        base_values=np.full(len(shap_values), explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[0]),
        data=X_top10.values,
        feature_names=predictor_features
    ),
    show=False,
    max_display=15
)
plt.title("SHAP Beeswarm - Top 10 Fraudulent Claims", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/shap_plots/beeswarm_top10.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/shap_plots/beeswarm_top10.png")

# Save detailed explanation
print("\n7. Saving detailed explanation...")
explanation_df = pd.DataFrame({
    'feature': predictor_features,
    'value': X_top10.iloc[0].values,
    'shap_value': shap_values[0],
    'abs_shap': np.abs(shap_values[0])
}).sort_values('abs_shap', ascending=False)

explanation_df.to_csv("results/shap_plots/top_claim_explanation.csv", index=False)
print("   ✓ Saved: results/shap_plots/top_claim_explanation.csv")

# Print summary
print("\n" + "="*80)
print("SHAP ANALYSIS COMPLETE!")
print("="*80)

print("\n📊 Top 5 Features Driving Fraud Score:")
print(f"{'Feature':<30} {'Value':<15} {'SHAP Impact':>15}")
print("-" * 62)
for _, row in explanation_df.head(5).iterrows():
    impact = "+" if row['shap_value'] > 0 else "-"
    print(f"{row['feature']:<30} {row['value']:<15.2f} {impact}{abs(row['shap_value']):>14.2f}")

print("\n📁 Generated Files:")
print("   • results/shap_plots/waterfall_top1.png      - Waterfall chart")
print("   • results/shap_plots/force_plot_top1.png     - Force plot")
print("   • results/shap_plots/bar_plot_top1.png       - Bar chart")
print("   • results/shap_plots/summary_top10.png       - Summary plot")
print("   • results/shap_plots/beeswarm_top10.png      - Beeswarm plot")
print("   • results/shap_plots/top_claim_explanation.csv - Detailed data")

print("\n💡 To view:")
print("   open results/shap_plots/waterfall_top1.png")
print("   open results/shap_plots/force_plot_top1.png")

print("\n✓ Done!\n")
