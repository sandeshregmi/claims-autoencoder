# 🎯 Databricks Bundle - Quick Reference Card

## 📁 Files Created

```
claims-autoencoder/
├── databricks.yml                       # Main bundle config
├── resources/
│   └── jobs/
│       ├── training_job.yml            # ML training pipeline (7 tasks)
│       ├── scoring_job.yml             # Batch scoring (3 tasks)
│       └── monitoring_job.yml          # Monitoring (4 tasks)
├── DATABRICKS_SETUP.md                 # Detailed setup guide
├── DATABRICKS_GITHUB_ARCHITECTURE.md   # Architecture documentation
├── DATABRICKS_IMPLEMENTATION_PLAN.md   # Week-by-week plan
└── validate_bundle.py                  # Validation script
```

## ⚡ Quick Commands

### Initial Setup
```bash
# Install Databricks CLI
pip install databricks-cli

# Configure (interactive)
databricks configure --token

# Validate bundle
databricks bundle validate

# Run validation script
python3 validate_bundle.py
```

### Deployment
```bash
# Deploy to DEV
databricks bundle deploy --target dev

# Deploy to STAGING
databricks bundle deploy --target staging

# Deploy to PROD
databricks bundle deploy --target prod
```

### Running Jobs
```bash
# Run training job
databricks bundle run model_training_job --target dev

# Run scoring job
databricks bundle run batch_scoring_job --target dev

# Run monitoring job
databricks bundle run monitoring_job --target dev
```

### Monitoring
```bash
# List jobs
databricks jobs list | grep "Claims Fraud"

# List recent runs
databricks runs list --limit 10

# Get run details
databricks runs get --run-id <RUN_ID>

# Get run output
databricks runs get-output --run-id <RUN_ID>

# Cancel run
databricks runs cancel --run-id <RUN_ID>
```

## 🏗️ Pipeline Architecture

### Training Pipeline (7 tasks, ~2 hours)
```
Data Ingestion (30min)
    ↓
Feature Engineering (20min)
    ↓
┌───────────────────────────┐
│ Train CatBoost (1hr)      │ ← Parallel
│ Train XGBoost (1hr)       │ ← Parallel
└───────────────────────────┘
    ↓
Model Evaluation (15min)
    ↓
┌────────────────────────────┐
│ Fairness Analysis (15min)  │ ← Parallel
│ PSI Monitoring (10min)     │ ← Parallel
└────────────────────────────┘
    ↓
Model Registration (10min)
```

### Key Features
- ✅ **Parallel training** of CatBoost and XGBoost
- ✅ **Automated fairness validation** before registration
- ✅ **PSI drift detection** to trigger retraining
- ✅ **MLflow integration** for experiment tracking
- ✅ **Auto-scaling clusters** for cost optimization
- ✅ **Spot instances** (60-90% cost savings)

## 🎛️ Configuration Variables

### Per-Environment Variables
```yaml
# databricks.yml

dev:
  catalog: dev_fraud_detection
  model_name: fraud_detector_dev
  cluster_max_workers: 2              # Smaller for dev
  schedule: PAUSED                    # Manual triggers only

staging:
  catalog: staging_fraud_detection
  model_name: fraud_detector_staging
  cluster_max_workers: 4              # Medium
  schedule: UNPAUSED                  # Runs on schedule

prod:
  catalog: prod_fraud_detection
  model_name: fraud_detector_prod
  cluster_max_workers: 8              # Larger
  schedule: UNPAUSED                  # Full automation
```

### Override at Deploy Time
```bash
databricks bundle deploy --target dev \
  --var catalog=my_custom_catalog \
  --var notification_email=me@example.com
```

## 📊 Job Schedules

| Job | Schedule | Environment |
|-----|----------|-------------|
| Training | Daily 2 AM | staging, prod |
| Scoring | Every 6 hours | staging, prod |
| Monitoring | Every 4 hours | staging, prod |
| All | Manual only | dev |

## 💰 Cost Estimates

### Development (per month)
- Training: ~$90-150 (daily runs)
- Scoring: ~$120 (4x daily)
- Monitoring: ~$90 (6x daily)
- **Total**: ~$300-360/month

### Production (per month)
- Training: ~$150-250
- Scoring: ~$200-300
- Monitoring: ~$150-200
- **Total**: ~$500-750/month

### Cost Optimization Tips
✅ Already configured:
- Spot instances (60-90% savings)
- Auto-termination (10-15 min idle)
- Auto-scaling (min 1-2, max 4-8)

## 🔐 Secrets Management

### Create Secret Scope
```bash
databricks secrets create-scope --scope fraud-detection
```

### Add Secrets
```bash
databricks secrets put --scope fraud-detection \
  --key api-token

databricks secrets put --scope fraud-detection \
  --key notification-webhook
```

### Use in Jobs
```yaml
environment_variables:
  API_TOKEN: {{secrets/fraud-detection/api-token}}
```

## 🚨 Troubleshooting

### Issue: Bundle validation fails
```bash
# Check YAML syntax
yamllint databricks.yml

# Run validation script with details
python3 validate_bundle.py
```

### Issue: Job fails with "File not found"
```bash
# Check file sync
databricks bundle deploy --target dev --verbose

# Files must be in src/ directory
# Paths are relative to bundle root
```

### Issue: Cluster creation fails
```bash
# Check node type for your region
# AWS: i3.xlarge, i3.2xlarge
# Azure: Standard_D8s_v3
# GCP: n1-highmem-8

# Update in databricks.yml:
cluster_node_type: "i3.xlarge"  # AWS
```

### Issue: Permission denied
```bash
# Check token has workspace access
databricks workspace ls /

# For prod, use service principal
run_as:
  service_principal_name: ${var.prod_service_principal}
```

## 📈 Monitoring Dashboard

Access in Databricks:
```
1. Jobs: Workspace → Workflows → Jobs
2. MLflow: Workspace → Machine Learning → Experiments
3. Unity Catalog: Data → Catalog Explorer
4. Alerts: Workspace → Workflows → Alerts
```

## 🔄 CI/CD Integration

### GitHub Actions Secrets Needed
```
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN_DEV=dapi...
DATABRICKS_TOKEN_STAGING=dapi...
DATABRICKS_TOKEN_PROD=dapi...
SLACK_WEBHOOK=https://hooks.slack.com/...
```

### Deploy from GitHub Actions
```yaml
- name: Deploy to DEV
  env:
    DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
    DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN_DEV }}
  run: |
    databricks bundle deploy --target dev
```

## ✅ Pre-Deployment Checklist

Before deploying to production:

- [ ] `python3 validate_bundle.py` passes
- [ ] Deployed to DEV successfully
- [ ] Training job completes without errors
- [ ] Fairness analysis validates
- [ ] PSI monitoring detects drift properly
- [ ] Model registers in MLflow
- [ ] Scoring job produces predictions
- [ ] Alerts trigger correctly
- [ ] Costs reviewed and approved
- [ ] GitHub Actions workflows tested

## 📚 Resources

- **Setup Guide**: `DATABRICKS_SETUP.md`
- **Architecture**: `DATABRICKS_GITHUB_ARCHITECTURE.md`
- **Implementation Plan**: `DATABRICKS_IMPLEMENTATION_PLAN.md`
- **Databricks Docs**: https://docs.databricks.com/dev-tools/bundles/
- **Jobs API**: https://docs.databricks.com/api/workspace/jobs
- **MLflow Guide**: https://docs.databricks.com/mlflow/

## 🎯 Next Steps

```bash
# 1. Validate everything
python3 validate_bundle.py

# 2. Deploy to DEV
databricks bundle deploy --target dev

# 3. Run training job
databricks bundle run model_training_job --target dev

# 4. Monitor in Databricks UI
# Go to: Workspace → Workflows → Jobs

# 5. Check results in MLflow
# Go to: Workspace → Machine Learning → Experiments
```

## 💡 Pro Tips

1. **Use validation script before every deploy**
   ```bash
   python3 validate_bundle.py && databricks bundle deploy --target dev
   ```

2. **Monitor job costs**
   ```bash
   # Check cluster usage
   databricks clusters list --output json | jq '.clusters[] | {name, state}'
   ```

3. **Test changes in DEV first**
   ```bash
   # Always test in dev before staging/prod
   databricks bundle deploy --target dev
   ```

4. **Use bundle variables for flexibility**
   ```bash
   # Override at deploy time
   databricks bundle deploy --target dev --var cluster_max_workers=1
   ```

5. **Version your deployments**
   ```bash
   # Tag releases
   git tag -a v1.0.0 -m "Production release"
   git push --tags
   ```

---

**Ready to deploy?** Run `python3 validate_bundle.py` to check everything! 🚀
