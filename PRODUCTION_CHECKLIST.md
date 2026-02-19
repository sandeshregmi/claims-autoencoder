# Production Deployment Checklist

✅ **Complete this checklist before deploying to production.**

---

## 📋 Pre-Deployment Tasks

### 1. Code Review & Quality

- [ ] Run tests: `make test-cov`
- [ ] Check code quality: `make lint`
- [ ] Format code: `make format`
- [ ] Review test coverage (target: >80%)
  ```bash
  make test-cov
  open htmlcov/index.html  # Review coverage report
  ```

### 2. Documentation

- [ ] Update `README_PRODUCTION.md` with your specific details
  - [ ] Replace placeholder URLs
  - [ ] Add your team contact info
  - [ ] Document custom configurations
- [ ] Create `CHANGELOG.md` if not present
- [ ] Verify all code has docstrings
  ```bash
  make lint  # Checks docstrings
  ```

### 3. Configuration

- [ ] Review `config/config.yaml`
  - [ ] Verify all paths are correct
  - [ ] Check model hyperparameters
  - [ ] Validate feature lists
- [ ] Create `.env.production` with production secrets
- [ ] Ensure `.gitignore` excludes `.env*` and sensitive files

### 4. Security

- [ ] No hardcoded credentials in code
- [ ] API keys/tokens in `.env` not `.env.example`
- [ ] Database credentials secured
- [ ] Review dependencies: `pip list`
  - [ ] No known vulnerabilities
  - [ ] All pinned to specific versions in `requirements.txt`

### 5. Dependencies

- [ ] Run: `make requirements`
- [ ] Verify `requirements.txt` is up-to-date
- [ ] Test clean install: 
  ```bash
  pip install -r requirements.txt
  ```
- [ ] All required packages available

### 6. Data Preparation

- [ ] Training data validated: `data/claims_train.parquet` exists
- [ ] Data quality checks passed
- [ ] No missing values in critical columns
- [ ] Feature distributions understood

### 7. Model Training

- [ ] Model trained and validated: `models/` directory populated
- [ ] Baseline performance documented
  ```bash
  # Train model
  claims-fraud train --config config/config.yaml
  ```
- [ ] Fraud detection rate meets threshold (>70%)
- [ ] False positive rate acceptable (<10%)
- [ ] Model reproducible with same random seed

---

## 🚢 Deployment Tasks

### 8. Build & Packaging

- [ ] Clean build: `make clean`
- [ ] Build package: `make build`
- [ ] Verify wheel created: `ls -lh dist/`
- [ ] Test package install:
  ```bash
  pip install dist/claims-fraud-*.whl
  claims-fraud --help
  ```

### 9. Local Testing

- [ ] Start dashboard locally: `make serve`
- [ ] Test all dashboard tabs work
- [ ] Scoring produces consistent results
- [ ] No errors in terminal output
- [ ] Load time acceptable

### 10. Databricks Setup (if applicable)

- [ ] Install Databricks CLI: `pip install databricks-cli`
- [ ] Configure Databricks: `databricks configure --token`
- [ ] Test connection: `make dbc-status`
- [ ] Create Databricks workspace if needed

### 11. Bundle Validation

- [ ] Validate bundle: `make validate`
  ```bash
  $ databricks bundle validate
  Validation OK!
  ```
- [ ] Review `databricks.yml`:
  - [ ] Correct workspace URL
  - [ ] Correct dev/prod targets
  - [ ] Correct artifact paths

### 12. DEV Deployment

- [ ] Deploy to DEV: `make deploy-dev`
- [ ] Monitor deployment logs
- [ ] Test DEV instance works
- [ ] Verify scores match local runs
- [ ] Run acceptance tests against DEV

### 13. Performance Testing

- [ ] Batch scoring speed acceptable (<100ms per 100 claims)
- [ ] Memory usage within limits
- [ ] No memory leaks on long-running processes
- [ ] Dashboard responsive (loads within 3s)

### 14. Monitoring Setup

- [ ] PSI monitoring configured
- [ ] Alert thresholds set (PSI > 0.25)
- [ ] Fairness monitoring enabled
- [ ] Logging to centralized system (if applicable)

---

## ⚠️ Production Deployment (PROD)

### 15. Pre-Production Sign-Off

- [ ] Product owner approves
- [ ] Data owner approves
- [ ] Security review passed
- [ ] Compliance review passed (if required)
- [ ] Stakeholders notified

### 16. PROD Deployment

```bash
# This requires confirmation
make deploy-prod
```

- [ ] Backup production data (if applicable)
- [ ] Schedule maintenance window
- [ ] Deploy to PROD
- [ ] Monitor deployment logs
- [ ] Verify PROD instance accessible

### 17. Post-Deployment Validation

- [ ] PROD dashboard accessible
- [ ] Scoring produces results
- [ ] Fairness metrics within acceptable range
- [ ] No errors in logs
- [ ] Performance metrics normal

### 18. Cleanup (After Successful Deployment)

Only after PROD is stable for 24+ hours:

- [ ] Archive old files: `make backup-old`
- [ ] Delete backup files (see `FILES_TO_DELETE.md`)
- [ ] Update repository with clean code
- [ ] Document deployment in `CHANGELOG.md`

---

## 📊 Post-Deployment Monitoring

### Week 1 Monitoring

- [ ] Check dashboard usage
- [ ] Monitor error logs daily
- [ ] Verify fraud scores reasonable
- [ ] Track model performance

### Week 2-4 Monitoring

- [ ] Monitor PSI trends
- [ ] Check fairness metrics
- [ ] Review model predictions
- [ ] Gather user feedback

### Ongoing (Monthly)

- [ ] Retrain model with new data
- [ ] Monitor data drift
- [ ] Update fairness analysis
- [ ] Review feature importance changes

---

## 🚨 Rollback Plan

If issues occur in PROD:

1. **Immediate**: Scale down model serving
2. **Within 1 hour**: Identify root cause
3. **Within 2 hours**: Deploy rollback or fix
4. **Communication**: Notify stakeholders

```bash
# Rollback to previous version
databricks bundle deploy --target prod --var version="0.0.1"
```

---

## 📝 Deployment Log

Fill this out during deployment:

```
Date: ________________
Deployed by: ________________
Version: 0.1.0
Environment: [ ] DEV  [ ] PROD

Pre-deployment checklist: [✓] Complete
Testing results: ________________
Issues encountered: ________________
Resolution: ________________
Approval from: ________________
Deployment time: ________________
Deployment status: [✓] Successful [ ] Failed [ ] Partial

Post-deployment validation: [✓] Passed
Rollback plan reviewed: [✓] Yes [ ] No
Stakeholders notified: [✓] Yes [ ] No
```

---

## 📞 Support Contacts

Update these with your team:

```
Product Owner: ________________ (email: _____)
Data Owner: ________________ (email: _____)
ML Engineer: ________________ (email: _____)
DevOps: ________________ (email: _____)
Escalation: ________________ (email: _____)
```

---

## ✅ Final Checklist

**Before clicking "Deploy":**

- [ ] All checklist items above completed
- [ ] Stakeholder approval obtained
- [ ] Rollback plan reviewed
- [ ] Support contacts documented
- [ ] Monitoring configured
- [ ] Team notified of deployment time

---

**Status:** Ready for production ✅

**Last Updated:** February 15, 2026  
**Next Review:** ________________
