"""
Complete SHAP Visualizer - ALL Plot Types Including Native SHAP Force Plot
Run this to generate Force, Waterfall, Bar, and Decision plots

Usage:
    python shap_all_plots_v2.py --claim_idx 0 --target claim_amount
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
    
    # Get SHAP values array
    if shap_values.ndim > 1:
        sv = shap_values[0]
    else:
        sv = shap_values
    
    # ═══════════════════════════════════════════════
    # 1. WATERFALL PLOT (Using SHAP's native waterfall)
    # ═══════════════════════════════════════════════
    print("  🌊 Creating Waterfall Plot (SHAP native)...")
    
    try:
        # Create SHAP Explanation object
        shap_explanation = shap.Explanation(
            values=sv,
            base_values=expected_value,
            data=X_claim.iloc[0].values,
            feature_names=predictor_features
        )
        
        # Generate waterfall plot
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_explanation, max_display=15, show=False)
        
        waterfall_path = output_dir / f"waterfall_claim{claim_idx}_{target_feature}.png"
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"     ✅ Saved: {waterfall_path}")
    except Exception as e:
        print(f"     ⚠️  Native waterfall failed: {e}")
        print(f"     Creating custom waterfall instead...")
        
        # Fallback to custom waterfall
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
    
    # ═══════════════════════════════════════════════
    # 2. FORCE PLOT (Using SHAP's native force plot)
    # ═══════════════════════════════════════════════
    print("  💪 Creating Force Plot (SHAP native)...")
    
    try:
        # Generate SHAP force plot
        shap.initjs()  # Initialize JavaScript
        
        # Create force plot
        force_plot = shap.force_plot(
            expected_value,
            sv,
            X_claim.iloc[0],
            feature_names=predictor_features,
            matplotlib=True,
            show=False
        )
        
        force_path = output_dir / f"force_claim{claim_idx}_{target_feature}.png"
        plt.savefig(force_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"     ✅ Saved: {force_path}")
        
        # Also save HTML version
        force_html = shap.force_plot(
            expected_value,
            sv,
            X_claim.iloc[0],
            feature_names=predictor_features,
            show=False
        )
        
        force_html_path = output_dir / f"force_claim{claim_idx}_{target_feature}.html"
        shap.save_html(str(force_html_path), force_html)
        print(f"     ✅ Saved: {force_html_path}")
        
    except Exception as e:
        print(f"     ⚠️  Native force plot failed: {e}")
        print(f"     Creating custom force plot...")
        
        # Custom force plot
        fig, ax = plt.subplots(figsize=(14, 4))
        
        # Sort by absolute value
        abs_shap = np.abs(sv)
        sorted_idx = np.argsort(abs_shap)[::-1][:10]
        
        # Create stacked bar
        base = expected_value
        cumsum = base
        
        for idx in sorted_idx:
            shap_val = sv[idx]
            feature_name = predictor_features[idx]
            color = '#d62728' if shap_val > 0 else '#1f77b4'
            
            ax.barh(0, shap_val, left=cumsum, height=0.6, 
                   color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
                   label=f"{feature_name}: {shap_val:+.3f}")
            cumsum += shap_val
        
        # Add markers
        ax.axvline(base, color='gray', linestyle='--', linewidth=2, 
                  label=f'Base: {base:.3f}', zorder=10)
        ax.axvline(cumsum, color='green', linestyle='--', linewidth=2, 
                  label=f'Prediction: {cumsum:.3f}', zorder=10)
        
        ax.set_xlabel('Feature Value Impact', fontsize=12, fontweight='bold')
        ax.set_title(f'SHAP Force Plot: {target_feature} (Claim {claim_idx})', 
                    fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.set_ylabel('')
        ax.grid(axis='x', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        force_path = output_dir / f"force_claim{claim_idx}_{target_feature}.png"
        plt.tight_layout()
        plt.savefig(force_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"     ✅ Saved: {force_path}")
    
    # ═══════════════════════════════════════════════
    # 3. BAR PLOT (Feature Importance)
    # ═══════════════════════════════════════════════
    print("  📊 Creating Bar Plot...")
    
    try:
        # Using SHAP's native bar plot
        plt.figure(figsize=(10, 8))
        
        shap_explanation = shap.Explanation(
            values=sv,
            base_values=expected_value,
            data=X_claim.iloc[0].values,
            feature_names=predictor_features
        )
        
        shap.plots.bar(shap_explanation, max_display=20, show=False)
        
        bar_path = output_dir / f"bar_claim{claim_idx}_{target_feature}.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"     ✅ Saved: {bar_path}")
        
    except Exception as e:
        print(f"     ⚠️  Native bar plot failed: {e}")
        print(f"     Creating custom bar plot...")
        
        # Custom bar plot
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
    
    try:
        # Using SHAP's native decision plot
        plt.figure(figsize=(10, 8))
        
        shap.decision_plot(
            expected_value,
            sv.reshape(1, -1),
            predictor_features,
            show=False,
            highlight=0
        )
        
        decision_path = output_dir / f"decision_claim{claim_idx}_{target_feature}.png"
        plt.tight_layout()
        plt.savefig(decision_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"     ✅ Saved: {decision_path}")
        
    except Exception as e:
        print(f"     ⚠️  Native decision plot failed: {e}")
        print(f"     Creating custom decision plot...")
        
        # Custom decision plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort by SHAP value
        sorted_idx = np.argsort(np.abs(sv))
        sorted_features = [predictor_features[i] for i in sorted_idx]
        sorted_shap = sv[sorted_idx]
        
        # Calculate cumulative sum
        cumsum = np.cumsum(sorted_shap)
        cumsum = np.insert(cumsum, 0, expected_value)
        
        # Plot
        y_pos = np.arange(len(sorted_features) + 1)
        ax.plot(cumsum, y_pos, 'o-', linewidth=2, markersize=6, color='steelblue')
        
        # Add feature names
        ax.set_yticks(y_pos[1:])
        ax.set_yticklabels(sorted_features, fontsize=8)
        ax.set_xlabel('Model Output', fontsize=12, fontweight='bold')
        ax.set_title(f'Decision Plot: {target_feature} (Claim {claim_idx})', 
                    fontsize=14, fontweight='bold')
        ax.axvline(expected_value, color='gray', linestyle='--', 
                  alpha=0.5, label=f'Base: {expected_value:.3f}')
        ax.grid(axis='x', alpha=0.3)
        ax.legend()
        
        decision_path = output_dir / f"decision_claim{claim_idx}_{target_feature}.png"
        plt.tight_layout()
        plt.savefig(decision_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"     ✅ Saved: {decision_path}")
    
    print(f"\n✅ All plots saved to: {output_dir}/")
    print("\n📊 Files created:")
    for file in sorted(output_dir.glob(f"*claim{claim_idx}_{target_feature}*")):
        print(f"   • {file.name}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate ALL SHAP plot types")
    parser.add_argument('--claim_idx', type=int, default=0, help="Claim index to analyze")
    parser.add_argument('--target', type=str, default='claim_amount', help="Target feature")
    parser.add_argument('--model_type', type=str, default='catboost', choices=['catboost', 'xgboost'])
    parser.add_argument('--config', type=str, default='config/example_config.yaml')
    
    args = parser.parse_args()
    
    print("🚀 Complete SHAP Visualizer v2")
    print("=" * 60)
    print("Includes: Waterfall, Force, Bar, and Decision plots")
    print("=" * 60)
    
    # Load config
    print("\n📋 Loading configuration...")
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # Load data
    print("📊 Loading data...")
    data_ingestion = DataIngestion(config)
    train_df, val_df, test_df = data_ingestion.load_train_val_test()
    
    print(f"   ✅ Loaded {len(val_df)} claims")
    
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
    
    print("\n" + "=" * 60)
    print("🎉 COMPLETE! Open the files:")
    print("=" * 60)
    print(f"\n   📂 cd {output_dir}")
    print(f"   🌐 open *.html")
    print(f"   🖼️  open *.png")
    print("\n💪 Force plot files:")
    force_files = list(output_dir.glob(f"force*claim{args.claim_idx}*"))
    for f in force_files:
        print(f"   • {f.name}")


if __name__ == "__main__":
    main()
