"""
Complete SHAP Visualizer - ALL Plot Types
Run this to generate Force, Waterfall, Bar, and Decision plots

Usage:
    python shap_all_plots.py --claim_idx 0 --target claim_amount
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import shap
    from shap_explainer import ClaimsShapExplainer
    from src.config_manager import ConfigManager
    from src.data_ingestion import DataIngestion
    from src.tree_models import ClaimsTreeAutoencoder
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the claims-autoencoder directory")
    sys.exit(1)


def create_all_plots(claim_idx, target_feature, model, data, explainer):
    """Generate ALL SHAP plot types for a single claim."""
    
    print(f"\n🔬 Generating SHAP plots for claim {claim_idx}, target: {target_feature}")
    
    claim_data = data.iloc[[claim_idx]]
    
    # Compute SHAP values
    print("  Computing SHAP values...")
    shap_values, contributions = explainer.explain_claim(claim_data, target_feature, plot=False)
    
    # Get explainer info
    explainer_info = explainer.explainers[target_feature]
    predictor_features = explainer_info['predictor_features']
    shap_explainer = explainer_info['explainer']
    
    # Prepare data
    X_claim = claim_data[predictor_features].copy()
    X_claim = explainer._preprocess_for_model(X_claim, predictor_features)
    
    # Get expected value
    expected_value = shap_explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value[0]
    
    # Create output directory
    output_dir = Path("shap_output")
    output_dir.mkdir(exist_ok=True)
    
    # ═══════════════════════════════════════════════
    # 1. WATERFALL PLOT
    # ═══════════════════════════════════════════════
    print("  📊 Creating Waterfall Plot...")
    
    top_contrib = contributions.head(15)
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_contrib['shap_value']]
    
    fig = go.Figure(go.Bar(
        y=top_contrib['feature'],
        x=top_contrib['shap_value'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in top_contrib['shap_value']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"SHAP Waterfall: {target_feature} (Claim {claim_idx})",
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        height=600
    )
    fig.add_vline(x=0, line_color='black', line_width=1.5)
    fig.update_yaxes(autorange="reversed")
    
    waterfall_path = output_dir / f"waterfall_claim{claim_idx}_{target_feature}.html"
    fig.write_html(str(waterfall_path))
    print(f"     ✅ Saved: {waterfall_path}")
    
    # ═════════════════════════════════════════════
    # 2. FORCE PLOT
    # ═══════════════════════════════════════════════
    print("  💪 Creating Force Plot...")
    
    fig, ax = plt.subplots(figsize=(14, 3))
    
    if shap_values.ndim > 1:
        sv = shap_values[0]
    else:
        sv = shap_values
    
    # Sort by absolute value
    abs_shap = np.abs(sv)
    sorted_idx = np.argsort(abs_shap)[::-1][:10]
    
    # Create force plot
    base = expected_value
    cumsum = base
    
    for idx in sorted_idx:
        shap_val = sv[idx]
        color = '#d62728' if shap_val > 0 else '#1f77b4'
        ax.barh(0, shap_val, left=cumsum, height=0.5, 
               color=color, alpha=0.7, edgecolor='black')
        cumsum += shap_val
    
    ax.axvline(base, color='gray', linestyle='--', linewidth=2, label=f'Base: {base:.3f}')
    ax.axvline(cumsum, color='green', linestyle='--', linewidth=2, label=f'Prediction: {cumsum:.3f}')
    
    ax.set_xlabel('Feature Value Impact', fontsize=12)
    ax.set_title(f'Force Plot: {target_feature} (Claim {claim_idx})', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    force_path = output_dir / f"force_claim{claim_idx}_{target_feature}.png"
    plt.tight_layout()
    plt.savefig(force_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     ✅ Saved: {force_path}")
    
    # ═══════════════════════════════════════════════
    # 3. BAR PLOT (Absolute SHAP)
    # ═══════════════════════════════════════════════
    print("  📊 Creating Bar Plot...")
    
    abs_contrib = contributions.copy()
    abs_contrib = abs_contrib.sort_values('abs_shap', ascending=False).head(20)
    
    fig = go.Figure(go.Bar(
        x=abs_contrib['abs_shap'],
        y=abs_contrib['feature'],
        orientation='h',
        marker_color=abs_contrib['abs_shap'],
        marker_colorscale='Reds',
        text=[f"{v:.4f}" for v in abs_contrib['abs_shap']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Feature Importance (|SHAP|): {target_feature} (Claim {claim_idx})",
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
        height=600
    )
    fig.update_yaxes(autorange="reversed")
    
    bar_path = output_dir / f"bar_claim{claim_idx}_{target_feature}.html"
    fig.write_html(str(bar_path))
    print(f"     ✅ Saved: {bar_path}")
    
    # ═══════════════════════════════════════════════
    # 4. DECISION PLOT
    # ═══════════════════════════════════════════════
    print("  🎯 Creating Decision Plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if shap_values.ndim > 1:
        sv = shap_values[0]
    else:
        sv = shap_values
    
    # Sort by SHAP value
    sorted_idx = np.argsort(np.abs(sv))
    sorted_features = [predictor_features[i] for i in sorted_idx]
    sorted_shap = sv[sorted_idx]
    
    # Calculate cumulative sum
    cumsum = np.cumsum(sorted_shap)
    cumsum = np.insert(cumsum, 0, expected_value)
    
    # Plot
    y_pos = np.arange(len(sorted_features) + 1)
    ax.plot(cumsum, y_pos, 'o-', linewidth=2, markersize=8, color='steelblue')
    
    # Add feature names
    ax.set_yticks(y_pos[1:])
    ax.set_yticklabels(sorted_features, fontsize=9)
    ax.set_xlabel('Model Output', fontsize=12)
    ax.set_title(f'Decision Plot: {target_feature} (Claim {claim_idx})', fontsize=14, fontweight='bold')
    ax.axvline(expected_value, color='gray', linestyle='--', alpha=0.5, label='Base value')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    
    decision_path = output_dir / f"decision_claim{claim_idx}_{target_feature}.png"
    plt.tight_layout()
    plt.savefig(decision_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     ✅ Saved: {decision_path}")
    
    print(f"\n✅ All plots saved to: {output_dir}/")
    print("\nFiles created:")
    print(f"  🌊 {waterfall_path.name}")
    print(f"  💪 {force_path.name}")
    print(f"  📊 {bar_path.name}")
    print(f"  🎯 {decision_path.name}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate ALL SHAP plot types")
    parser.add_argument('--claim_idx', type=int, default=0, help="Claim index to analyze")
    parser.add_argument('--target', type=str, default='claim_amount', help="Target feature")
    parser.add_argument('--model_type', type=str, default='catboost', choices=['catboost', 'xgboost'])
    parser.add_argument('--config', type=str, default='config/example_config.yaml')
    
    args = parser.parse_args()
    
    print("🚀 Complete SHAP Visualizer")
    print("=" * 50)
    
    # Load config
    print("\n📋 Loading configuration...")
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # Load data
    print("📊 Loading data...")
    data_ingestion = DataIngestion(config)
    train_df, val_df, test_df = data_ingestion.load_train_val_test()
    
    print(f"   Loaded {len(val_df)} claims")
    
    # Create and train model
    print(f"\n🤖 Training {args.model_type} model...")
    model = ClaimsTreeAutoencoder(model_type=args.model_type)
    model.fit(
        val_df,
        config.data.categorical_features,
        config.data.numerical_features,
        verbose=False
    )
    print("   ✅ Model trained")
    
    # Initialize SHAP
    print("\n🔬 Initializing SHAP explainer...")
    explainer = ClaimsShapExplainer(
        model,
        model.feature_names,
        config.data.categorical_features
    )
    explainer.create_explainers(val_df, max_samples=100)
    print("   ✅ SHAP ready")
    
    # Generate plots
    output_dir = create_all_plots(args.claim_idx, args.target, model, val_df, explainer)
    
    print("\n" + "=" * 50)
    print("🎉 COMPLETE! Open the HTML files in your browser:")
    print(f"\n   open {output_dir}/*.html")
    print(f"   open {output_dir}/*.png")


if __name__ == "__main__":
    main()
