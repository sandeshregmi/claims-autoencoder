# GitHub Actions Troubleshooting Guide

## Issue: GitHub Actions not running

### Quick Diagnostics

The workflow file `.github/workflows/ci.yml` is correctly configured with:
- ✅ Trigger on `push` to `main` and `develop` branches
- ✅ Trigger on `pull_request` to `main` and `develop` branches
- ✅ Valid YAML syntax
- ✅ Proper job configuration

### Common Causes & Solutions

#### 1. **GitHub Actions Not Enabled**
Check if Actions are enabled on your repository:
- Go to: https://github.com/sandeshregmi/claims-autoencoder/settings/actions
- Ensure "Actions permissions" is set to allow workflows

#### 2. **Branch Protection Rules**
If you have branch protection rules:
- Go to: https://github.com/sandeshregmi/claims-autoencoder/settings/branches
- Check if "Require status checks to pass before merging" is enabled
- Add the CI workflow checks to required status checks

#### 3. **Workflow Permissions**
Check workflow permissions:
- Go to: https://github.com/sandeshregmi/claims-autoencoder/settings/actions
- Scroll to "Workflow permissions"
- Ensure "Read and write permissions" is selected (if needed)

#### 4. **Manual Trigger**
To manually trigger the workflow:
```bash
# Push a change to main or develop
git commit --allow-empty -m "Trigger CI workflow"
git push origin main
```

#### 5. **Check Workflow Runs**
View workflow execution history:
- Go to: https://github.com/sandeshregmi/claims-autoencoder/actions
- Look for the "CI - Tests and Validation" workflow
- Click to see details and logs

### Workflow Status Check

The latest commits should trigger the workflow:
- Last commit: `a6a61e7` (HEAD -> main, origin/main)
- Remote: https://github.com/sandeshregmi/claims-autoencoder.git

### Logs Location
If the action runs, check logs at:
```
https://github.com/sandeshregmi/claims-autoencoder/actions/runs/[RUN_ID]
```

### Alternative: Enable Debug Logging
Add this to your workflow for verbose output:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
```

### Local Validation
You can validate workflow syntax locally:
```bash
# Install act (GitHub Actions runner simulator)
# Mac: brew install act

# Run workflow locally
act push -b
```

## Quick Links
- Actions Page: https://github.com/sandeshregmi/claims-autoencoder/actions
- Workflow File: https://github.com/sandeshregmi/claims-autoencoder/blob/main/.github/workflows/ci.yml
- Settings: https://github.com/sandeshregmi/claims-autoencoder/settings/actions

## Next Steps
1. Verify Actions are enabled
2. Check branch protection rules
3. View workflow execution logs
4. Manually trigger with a test commit if needed
