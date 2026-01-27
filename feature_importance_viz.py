"""
Feature Importance Visualization (No SHAP Required)
Creates charts showing what drives fraud scores using built-in model importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.tree_models import ClaimsTreeAutoencoder
from src.config_manager import ConfigManager
from src.data_ingestion import DataIngestion

print("\n" + "="*80)
print("FRAUD FEATURE IMPORTANCE ANALYZER")
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
fraud_scores, per_feature_errors = model.compute_fraud_scores(val_df)
print(f"   ✓ Scores computed (mean: {fraud_scores.mean():,.0f})")

# Get top fraudulent claims
top_10_indices = fraud_scores.argsort()[-10:][::-1]
print(f"\n4. Top 10 fraud scores: {fraud_scores[top_10_indices[0]]:,.0f} to {fraud_scores[top_10_indices[-1]]:,.0f}")

# Create output directory
output_dir = Path("results/feature_importance")
output_dir.mkdir(parents=True, exist_ok=True)

# Get feature importance
print("\n5. Computing feature importance...")
all_importances = model.get_feature_importance()

# Average importance across all targets
avg_importance = {}
for target_feat, importances in all_importances.items():
    for feat, imp in importances.items():
        if feat not in avg_importance:
            avg_importance[feat] = []
        avg_importance[feat].append(imp)

mean_importance = {feat: np.mean(imps) for feat, imps in avg_importance.items()}
importance_df = pd.DataFrame([
    {'feature': feat, 'importance': imp}
    for feat, imp in mean_importance.items()
]).sort_values('importance', ascending=False)

print("   ✓ Feature importance computed")

# PLOT 1: Overall Feature Importance
print("\n6. Creating visualizations...")
print("   Creating overall importance chart...")
plt.figure(figsize=(12, 8))
top_n = 15
top_features = importance_df.head(top_n)

colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
plt.barh(range(len(top_features)), top_features['importance'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.title('Top 15 Features for Fraud Detection', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "overall_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/feature_importance/overall_importance.png")

# PLOT 2: Feature Errors for Top Fraudulent Claim
print("   Creating top claim feature analysis...")
most_fraudulent_idx = top_10_indices[0]
claim_errors = {feat: errors[most_fraudulent_idx] for feat, errors in per_feature_errors.items()}
claim_error_df = pd.DataFrame([
    {'feature': feat, 'error': err}
    for feat, err in claim_errors.items()
]).sort_values('error', ascending=False).head(15)

plt.figure(figsize=(12, 8))
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(claim_error_df)))
plt.barh(range(len(claim_error_df)), claim_error_df['error'], color=colors)
plt.yticks(range(len(claim_error_df)), claim_error_df['feature'])
plt.xlabel('Reconstruction Error', fontsize=12, fontweight='bold')
plt.title(f'Feature Anomalies - Most Fraudulent Claim (Score: {fraud_scores[most_fraudulent_idx]:,.0f})', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "top_claim_anomalies.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/feature_importance/top_claim_anomalies.png")

# PLOT 3: Fraud Score Distribution
print("   Creating fraud score distribution...")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(fraud_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
plt.axvline(np.percentile(fraud_scores, 95), color='red', linestyle='--', 
            linewidth=2, label='95th percentile')
plt.axvline(np.percentile(fraud_scores, 99), color='darkred', linestyle='--', 
            linewidth=2, label='99th percentile')
plt.xlabel('Fraud Score', fontsize=11, fontweight='bold')
plt.ylabel('Number of Claims', fontsize=11, fontweight='bold')
plt.title('Fraud Score Distribution', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(np.log10(fraud_scores + 1), bins=50, color='coral', alpha=0.7, edgecolor='black')
plt.xlabel('Log10(Fraud Score)', fontsize=11, fontweight='bold')
plt.ylabel('Number of Claims', fontsize=11, fontweight='bold')
plt.title('Fraud Score Distribution (Log Scale)', fontsize=13, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "fraud_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/feature_importance/fraud_distribution.png")

# PLOT 4: Top 10 Claims Heatmap
print("   Creating top 10 claims heatmap...")
top_10_errors = []
for idx in top_10_indices:
    errors = [per_feature_errors[feat][idx] for feat in model.feature_names]
    top_10_errors.append(errors)

heatmap_data = np.array(top_10_errors).T
heatmap_df = pd.DataFrame(heatmap_data, 
                          index=model.feature_names,
                          columns=[f"#{i+1}" for i in range(10)])

# Normalize for better visualization
heatmap_normalized = (heatmap_df - heatmap_df.min()) / (heatmap_df.max() - heatmap_df.min())

plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_normalized, cmap='YlOrRd', cbar_kws={'label': 'Normalized Error'})
plt.title('Feature Anomalies Across Top 10 Fraudulent Claims', fontsize=14, fontweight='bold')
plt.xlabel('Claim Rank', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "top10_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/feature_importance/top10_heatmap.png")

# PLOT 5: Feature Contribution to Top Claim
print("   Creating waterfall-style chart...")
top_claim_data = val_df.iloc[most_fraudulent_idx]
claim_contributions = claim_error_df.copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left: Feature errors
ax1.barh(range(len(claim_contributions)), claim_contributions['error'], 
         color='crimson', alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(claim_contributions)))
ax1.set_yticklabels(claim_contributions['feature'])
ax1.set_xlabel('Error Magnitude', fontsize=12, fontweight='bold')
ax1.set_title('Feature Reconstruction Errors', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Right: Feature values
feature_values = []
feature_names_with_values = []
for feat in claim_contributions['feature']:
    val = top_claim_data.get(feat, 'N/A')
    if isinstance(val, (int, float)):
        feature_values.append(val)
        feature_names_with_values.append(f"{feat}\n({val:.1f})")
    else:
        feature_values.append(0)
        feature_names_with_values.append(f"{feat}\n({val})")

ax2.barh(range(len(feature_values)), feature_values, 
         color='steelblue', alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(feature_names_with_values)))
ax2.set_yticklabels(feature_names_with_values, fontsize=9)
ax2.set_xlabel('Feature Value', fontsize=12, fontweight='bold')
ax2.set_title('Actual Feature Values', fontsize=13, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

plt.suptitle(f'Detailed Analysis - Most Fraudulent Claim\n(Fraud Score: {fraud_scores[most_fraudulent_idx]:,.0f})', 
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "top_claim_detailed.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: results/feature_importance/top_claim_detailed.png")

# Save detailed data
print("\n7. Saving detailed data...")
importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
print("   ✓ Saved: results/feature_importance/feature_importance.csv")

# Save top 10 claims analysis
top_10_df = val_df.iloc[top_10_indices].copy()
top_10_df['fraud_score'] = fraud_scores[top_10_indices]
top_10_df['rank'] = range(1, 11)
cols = ['rank', 'fraud_score'] + [c for c in top_10_df.columns if c not in ['rank', 'fraud_score']]
top_10_df = top_10_df[cols]
top_10_df.to_csv(output_dir / "top_10_fraudulent_claims.csv", index=False)
print("   ✓ Saved: results/feature_importance/top_10_fraudulent_claims.csv")

# Print summary
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print("\n📊 Top 5 Most Important Features:")
for i, row in importance_df.head(5).iterrows():
    print(f"   {i+1}. {row['feature']:<35} {row['importance']:>10.4f}")

print("\n🚨 Most Fraudulent Claim Details:")
print(f"   Fraud Score: {fraud_scores[most_fraudulent_idx]:,.0f}")
print(f"   Claim Amount: ${top_claim_data.get('claim_amount', 'N/A'):,.2f}")
print(f"   Claim Type: {top_claim_data.get('claim_type', 'N/A')}")
print(f"   Previous Claims: {top_claim_data.get('num_previous_claims', 'N/A')}")

print("\n📊 Top Anomalous Features for This Claim:")
for i, row in claim_error_df.head(5).iterrows():
    val = top_claim_data.get(row['feature'], 'N/A')
    print(f"   {i+1}. {row['feature']:<25} Error: {row['error']:>10.2f}  Value: {val}")

print("\n📁 Generated Files:")
print("   • results/feature_importance/overall_importance.png")
print("   • results/feature_importance/top_claim_anomalies.png")
print("   • results/feature_importance/fraud_distribution.png")
print("   • results/feature_importance/top10_heatmap.png")
print("   • results/feature_importance/top_claim_detailed.png")
print("   • results/feature_importance/feature_importance.csv")
print("   • results/feature_importance/top_10_fraudulent_claims.csv")

print("\n💡 To view charts:")
print("   open results/feature_importance/top_claim_detailed.png")
print("   open results/feature_importance/overall_importance.png")

print("\n✓ Done!\n")
