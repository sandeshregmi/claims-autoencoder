# 🎯 Databricks Bundle Setup Guide

## ✅ What Was Created

Your Databricks Asset Bundle is now ready with:

1. **`databricks.yml`** - Main bundle configuration
2. **`resources/jobs/training_job.yml`** - Complete ML training pipeline
3. **`resources/jobs/scoring_job.yml`** - Batch scoring pipeline
4. **`resources/jobs/monitoring_job.yml`** - Drift & fairness monitoring

## 📋 Prerequisites

### 1. Databricks CLI Installation

```bash
# Install Databricks CLI
pip install databricks-cli

# Verify installation
databricks --version
```

### 2. Databricks Workspace Setup

You need:
- ✅ Databricks workspace URL
- ✅ Personal access token
- ✅ Unity Catalog enabled (optional but recommended)

### 3. Get Your Workspace URL

```
Example: https://dbc-a1b2c3d4-e5f6.cloud.databricks.com
```

### 4. Generate Access Token

1. Go to your Databricks workspace
2. Click on your profile → Settings
3. Go to Developer → Access tokens
4. Generate new token
5. **Save it securely!**

## 🚀 Quick Start

### Step 1: Configure Databricks CLI

```bash
# Interactive configuration
databricks configure --token

# You'll be prompted for:
# - Databricks Host: https://your-workspace.cloud.databricks.com
# - Token: [paste your token]
```

Or create `~/.databrickscfg` manually:

```ini
[DEFAULT]
host = https://your-workspace.cloud.databricks.com
token = dapi1234567890abcdef
```

### Step 2: Update databricks.yml

Edit `databricks.yml` and update:

```yaml
# Line 188-190: Update workspace host
workspace:
  host: https://YOUR-ACTUAL-WORKSPACE.cloud.databricks.com  # ⬅️ CHANGE THIS
```

### Step 3: Validate Bundle

```bash
# Navigate to project directory
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Validate configuration
databricks bundle validate

# Expected output:
# ✓ Configuration is valid
```

### Step 4: Deploy to DEV

```bash
# Deploy to development environment
databricks bundle deploy --target dev

# Expected output:
# ✓ Uploaded files
# ✓ Created jobs
# ✓ Deployment complete
```

### Step 5: View Deployed Resources

```bash
# List deployed jobs
databricks jobs list --output json | jq '.jobs[] | select(.settings.name | contains("Claims Fraud"))'

# Or in browser:
# Go to: Workspace > Workflows > Jobs
# You should see:
# - [dev] Claims Fraud - Model Training
# - [dev] Claims Fraud - Batch Scoring  
# - [dev] Claims Fraud - Monitoring
```

### Step 6: Run Your First Job

```bash
# Run training job
databricks bundle run model_training_job --target dev

# Monitor job status
databricks runs list --limit 1 --output json | jq '.[0]'
```

## 🔧 Configuration Options

### Environment Variables

You can set these in your shell or CI/CD:

```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi1234567890abcdef"
export DATABRICKS_CLUSTER_ID="1234-567890-abcd123"  # Optional
```

### Bundle Variables

Override variables per environment:

```bash
# Deploy with custom catalog
databricks bundle deploy --target dev --var catalog=my_custom_catalog

# Deploy with custom notification email
databricks bundle deploy --target dev --var notification_email=me@example.com
```

## 📊 Understanding the Pipeline

### Training Job Flow

```
1. Data Ingestion (30 min)
   ↓
2. Feature Engineering (20 min)
   ↓
3a. Train CatBoost (1 hr) ⟋  
3b. Train XGBoost (1 hr) ⟍ → 4. Model Evaluation (15 min)
                              ↓
                          5. Fairness Analysis (15 min)
                              ↓
                          6. PSI Monitoring (10 min)
                              ↓
                          7. Model Registration (10 min)
                          
Total time: ~2-2.5 hours
Cost: ~$5-10 per run (dev environment)
```

### What Gets Created

After deployment, you'll have:

```
Databricks Workspace/
├── .bundle/
│   └── claims-fraud-detection/
│       └── dev/
│           ├── files/
│           │   ├── src/
│           │   ├── config/
│           │   └── requirements.txt
│           └── state/
│               └── deployment.json
│
└── Jobs created:
    ├── [dev] Claims Fraud - Model Training
    ├── [dev] Claims Fraud - Batch Scoring
    └── [dev] Claims Fraud - Monitoring
```

## 🧪 Testing the Deployment

### Test 1: Validate Bundle

```bash
databricks bundle validate
```

**Expected**: ✓ Configuration is valid

### Test 2: Deploy to DEV

```bash
databricks bundle deploy --target dev
```

**Expected**: 
```
Uploading files...
Creating jobs...
✓ Deployment complete
```

### Test 3: List Jobs

```bash
databricks jobs list | grep "Claims Fraud"
```

**Expected**: Should show 3 jobs

### Test 4: Describe a Job

```bash
databricks jobs get --job-id <JOB_ID>
```

### Test 5: Run a Job

```bash
# Start training job
databricks bundle run model_training_job --target dev

# Check status
databricks runs list --limit 5
```

## 🐛 Troubleshooting

### Issue 1: "Invalid configuration"

**Symptom**: `databricks bundle validate` fails

**Solution**:
1. Check YAML syntax (indentation matters!)
2. Verify all required fields are present
3. Run: `yamllint databricks.yml`

### Issue 2: "Cluster not found"

**Symptom**: Job fails with cluster error

**Solution**:
- Job clusters are created automatically
- No existing cluster needed
- Check node_type_id matches your workspace region

### Issue 3: "Permission denied"

**Symptom**: Can't deploy bundle

**Solution**:
1. Verify your token has workspace access
2. Check you're not deploying to a protected path
3. For prod, use service principal

### Issue 4: "Python file not found"

**Symptom**: Task fails with "File not found"

**Solution**:
- Files must be in `src/` directory
- Path is relative to bundle root
- Run: `databricks bundle validate` to check sync config

## 📈 Next Steps

### 1. Setup Staging Environment

```bash
# Deploy to staging
databricks bundle deploy --target staging
```

### 2. Configure Secrets

```bash
# Create secret scope
databricks secrets create-scope --scope fraud-detection

# Add secrets
databricks secrets put --scope fraud-detection --key api-token
```

### 3. Setup GitHub Actions

See `.github/workflows/` for CI/CD pipelines

### 4. Monitor Jobs

```bash
# List recent runs
databricks runs list --limit 10

# Get run details
databricks runs get --run-id <RUN_ID>

# View logs
databricks runs get-output --run-id <RUN_ID>
```

### 5. Access MLflow

```
URL: https://your-workspace.cloud.databricks.com/#mlflow/experiments
```

## 💰 Cost Estimation

### Development Environment

**Training Job** (daily):
- Cluster: 2x i3.xlarge (spot)
- Duration: ~2 hours
- Cost: ~$3-5 per run
- Monthly: ~$90-150

**Scoring Job** (every 6 hours):
- Cluster: 2x i3.xlarge (spot)
- Duration: ~30 minutes
- Cost: ~$1 per run
- Monthly: ~$120

**Monitoring Job** (every 4 hours):
- Cluster: 1x i3.xlarge (spot)
- Duration: ~15 minutes
- Cost: ~$0.50 per run
- Monthly: ~$90

**Total DEV Monthly**: ~$300-360

### Production Environment

**With auto-scaling and optimizations**: ~$800-1200/month

**Tips to reduce costs**:
- Use spot instances (already configured)
- Schedule jobs during off-peak hours
- Enable auto-termination (already configured)
- Use smaller clusters for dev/staging

## 📚 Additional Resources

- [Databricks Bundles Documentation](https://docs.databricks.com/dev-tools/bundles/index.html)
- [Jobs API Reference](https://docs.databricks.com/api/workspace/jobs)
- [Unity Catalog Guide](https://docs.databricks.com/data-governance/unity-catalog/index.html)
- [MLflow on Databricks](https://docs.databricks.com/mlflow/index.html)

## ✅ Verification Checklist

Before deploying to production:

- [ ] Bundle validates successfully
- [ ] Deployed to DEV successfully
- [ ] Training job runs without errors
- [ ] Fairness analysis produces results
- [ ] PSI monitoring detects drift
- [ ] Model registers in MLflow
- [ ] Scoring job produces predictions
- [ ] Monitoring alerts trigger correctly
- [ ] GitHub Actions CI/CD works
- [ ] Costs are within budget

## 🎯 You're Ready!

Your Databricks Asset Bundle is configured and ready to deploy. Start with:

```bash
databricks bundle validate && databricks bundle deploy --target dev
```

Then monitor your first training run in the Databricks workspace! 🚀
