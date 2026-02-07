# Clean Workflow Pipeline - Visual Guide

## 🎯 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   CLEAN WORKFLOW PIPELINE                    │
│                  Claims Fraud Detection                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────┐
        │     1. DATA PREPARATION                 │
        │  ┌──────────────────────────────────┐  │
        │  │  data/claims_train.parquet       │  │
        │  │         │                         │  │
        │  │         ▼                         │  │
        │  │  DataIngestion                    │  │
        │  │  ├─ Load parquet                  │  │
        │  │  ├─ Split train/val/test          │  │
        │  │  └─ Validate schema               │  │
        │  └──────────────────────────────────┘  │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │     2. PREPROCESSING                    │
        │  ┌──────────────────────────────────┐  │
        │  │  ClaimsPreprocessor              │  │
        │  │  ├─ Handle missing values        │  │
        │  │  ├─ Encode categoricals          │  │
        │  │  ├─ Scale numerical features     │  │
        │  │  └─ Feature engineering          │  │
        │  └──────────────────────────────────┘  │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │     3. MODEL TRAINING                   │
        │  ┌──────────────────────────────────┐  │
        │  │  ClaimsTreeAutoencoder           │  │
        │  │  ├─ XGBoost or CatBoost          │  │
        │  │  ├─ Hyperparameter tuning        │  │
        │  │  ├─ Cross-validation             │  │
        │  │  └─ Save best model              │  │
        │  └──────────────────────────────────┘  │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │     4. EXPLAINABILITY                   │
        │  ┌──────────────────────────────────┐  │
        │  │  ClaimsShapExplainer             │  │
        │  │  ├─ Compute SHAP values          │  │
        │  │  ├─ Feature importance           │  │
        │  │  ├─ Force plots                  │  │
        │  │  └─ Dependence plots             │  │
        │  └──────────────────────────────────┘  │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │     5. MONITORING                       │
        │  ┌──────────────────────────────────┐  │
        │  │  PSIMonitor                      │  │
        │  │  ├─ Calculate PSI scores         │  │
        │  │  ├─ Detect drift                 │  │
        │  │  └─ Alert thresholds             │  │
        │  └──────────────────────────────────┘  │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │     6. FAIRNESS ANALYSIS                │
        │  ┌──────────────────────────────────┐  │
        │  │  FairnessAnalyzer                │  │
        │  │  ├─ Demographic parity           │  │
        │  │  ├─ Equal opportunity            │  │
        │  │  └─ Disparate impact             │  │
        │  └──────────────────────────────────┘  │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │     7. WEB DASHBOARD                    │
        │  ┌──────────────────────────────────┐  │
        │  │  webapp_enhanced.py              │  │
        │  │  ├─ Overview Tab                 │  │
        │  │  ├─ Predictions Tab              │  │
        │  │  ├─ Feature Importance Tab       │  │
        │  │  ├─ PSI Monitoring Tab           │  │
        │  │  ├─ Fairness Tab                 │  │
        │  │  └─ SHAP Analysis Tab            │  │
        │  └──────────────────────────────────┘  │
        └────────────────────────────────────────┘
```

## 📁 File Organization

```
claims-autoencoder/
│
├── 🎯 CORE MODULES
│   ├── src/webapp_enhanced.py         ⭐ Main application
│   ├── src/tree_models.py             🌳 Model training
│   ├── src/preprocessing.py           🔧 Data processing
│   ├── src/config_manager.py          ⚙️ Configuration
│   ├── src/data_ingestion.py         📊 Data loading
│   ├── src/psi_monitoring.py         📈 Drift detection
│   └── src/fairness_analysis.py      ⚖️ Fairness metrics
│
├── 🔍 EXPLAINABILITY
│   └── shap_explainer.py             🎨 SHAP analysis
│
├── ⚙️ CONFIGURATION
│   └── config/starter_config.yaml     📝 Settings
│
├── 📊 DATA
│   └── data/claims_train.parquet      💾 Training data
│
├── 🤖 MODELS
│   └── models/                        💼 Saved models
│
├── 🚀 EXECUTION
│   ├── run_clean_workflow.sh         ▶️ Main runner
│   └── clean_workflow.sh             🧹 Cleanup tool
│
├── 📦 DEPENDENCIES
│   └── requirements_clean.txt         📋 Minimal packages
│
└── 📚 DOCUMENTATION
    ├── README_CLEAN.md                📖 Main guide
    ├── CLEAN_WORKFLOW.md              🗺️ Architecture
    ├── CLEANUP_COMPLETE.md            ✅ Summary
    └── WORKFLOW_DIAGRAM.md            📊 This file
```

## 🔄 Data Flow

```
Input Data (Parquet)
        │
        ├─► Load & Validate
        │         │
        │         ▼
        │   Train/Val/Test Split
        │         │
        │         ├─► Training Set
        │         │         │
        │         │         ▼
        │         │   Preprocessing
        │         │         │
        │         │         ▼
        │         │   Model Training
        │         │         │
        │         │         ├─► XGBoost Model
        │         │         └─► CatBoost Model
        │         │                   │
        │         │                   ▼
        │         │            Best Model Saved
        │         │                   │
        │         ▼                   │
        │   Validation Set ───────────┤
        │         │                   │
        │         ▼                   ▼
        │   Performance Metrics   SHAP Values
        │         │                   │
        │         └──────┬────────────┘
        │                │
        ▼                ▼
  Test Set         Web Dashboard
        │                │
        ├─► Predictions  │
        ├─► PSI Scores   │
        └─► Fairness ────┘
```

## 🎨 Dashboard Tabs

```
┌─────────────────────────────────────────────┐
│          STREAMLIT DASHBOARD                │
├─────────────────────────────────────────────┤
│                                             │
│  📊 OVERVIEW                                │
│  ├─ Fraud Statistics                       │
│  ├─ Model Metrics                          │
│  └─ Visualizations                         │
│                                             │
│  🔮 PREDICTIONS                             │
│  ├─ Single Claim Scoring                   │
│  ├─ Batch Predictions                      │
│  └─ Risk Assessments                       │
│                                             │
│  ⭐ FEATURE IMPORTANCE                      │
│  ├─ Global Importance                      │
│  ├─ SHAP Waterfall                         │
│  └─ Feature Dependencies                   │
│                                             │
│  📈 PSI MONITORING                          │
│  ├─ Overall PSI Score                      │
│  ├─ Feature-Level PSI                      │
│  └─ Drift Alerts                           │
│                                             │
│  ⚖️ FAIRNESS ANALYSIS                       │
│  ├─ Demographic Parity                     │
│  ├─ Equal Opportunity                      │
│  └─ Disparate Impact                       │
│                                             │
│  🔬 SHAP ANALYSIS                           │
│  ├─ Force Plots                            │
│  ├─ Summary Plots                          │
│  ├─ Dependence Plots                       │
│  └─ Individual Explanations                │
│                                             │
└─────────────────────────────────────────────┘
```

## 🚀 Execution Flow

```
User runs: ./run_clean_workflow.sh
              │
              ├─► Check virtual environment
              │         │
              │         ├─ Exists? ─► Activate
              │         └─ No? ────► Create & Activate
              │
              ├─► Install dependencies
              │         │
              │         └─► pip install -r requirements_clean.txt
              │
              ├─► Verify files
              │         │
              │         ├─ Data exists?
              │         ├─ Config exists?
              │         └─ Source files exist?
              │
              ├─► Initialize application
              │         │
              │         ├─ Load configuration
              │         ├─ Initialize models
              │         └─ Setup logging
              │
              └─► Launch Streamlit
                        │
                        ▼
                  Browser opens
                        │
                        ▼
                  Dashboard at localhost:8501
```

## 🧹 Cleanup Process

```
User runs: ./clean_workflow.sh
              │
              ├─► Create backup directory
              │         │
              │         └─► backup_YYYYMMDD_HHMMSS/
              │
              ├─► Move documentation files
              │         │
              │         ├─ *_COMPLETE.md
              │         ├─ *_FIX.md
              │         ├─ *_GUIDE.md
              │         └─ ... (40+ files)
              │
              ├─► Move duplicate apps
              │         │
              │         ├─ app_complete.py
              │         ├─ app_enhanced.py
              │         └─ webapp_enhanced_*.py
              │
              ├─► Move utility scripts
              │         │
              │         ├─ add_*.py
              │         ├─ apply_*.py
              │         └─ ... (30+ files)
              │
              ├─► Remove cache directories
              │         │
              │         ├─ __pycache__/
              │         ├─ .pytest_cache/
              │         └─ catboost_info/
              │
              └─► Summary report
                        │
                        └─► Shows what was cleaned
```

## 📊 Component Relationships

```
                    ConfigManager
                         │
                         ├─── reads ───► starter_config.yaml
                         │
                    ┌────┴────┐
                    │         │
                    ▼         ▼
            DataIngestion  TreeModels
                    │         │
                    │         ├─── uses ───► Preprocessor
                    │         │
                    │         ├─── trains ─► XGBoost/CatBoost
                    │         │
                    │         └─── saves ──► models/
                    │
                    ▼
            ShapExplainer ───► generates ───► SHAP values
                    │
                    │
                    ▼
            PSIMonitor ──────► monitors ────► Data Drift
                    │
                    │
                    ▼
            FairnessAnalyzer ► checks ──────► Bias
                    │
                    │
                    ▼
            WebApp (Streamlit) ─────────────► Dashboard
                    │
                    └─── integrates all components
```

## 🎯 Key Benefits Visualization

```
BEFORE CLEANUP                    AFTER CLEANUP
─────────────────────────────────────────────
📁 150+ files                     📁 ~20 files
🗂️ Confusing structure            🎯 Clear structure  
❌ Multiple duplicates            ✅ Single source
📝 Outdated docs                  📚 Updated docs
🐛 Hard to maintain               🔧 Easy to maintain
⏱️ Slow onboarding               🚀 Quick start
```

---

**This visual guide shows the complete clean workflow architecture!** 📊
