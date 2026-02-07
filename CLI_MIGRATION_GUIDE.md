# 🔧 Databricks CLI Migration Guide

## Issue: "No such command 'bundle'"

You're seeing this error because you have the **old Databricks CLI** installed. Asset Bundles require the **new Databricks CLI**.

```
Error: No such command 'bundle'.
Warning: The version of the CLI you are using is deprecated.
```

---

## ✅ Solution: Install New Databricks CLI

### Option 1: Quick Install (Recommended)

```bash
# 1. Uninstall old CLI
pip uninstall databricks-cli

# 2. Install new CLI
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

# 3. Verify installation
databricks --version
# Should show: Databricks CLI v0.2xx.x or higher

# 4. Test bundle command
databricks bundle --help
# Should show bundle commands
```

### Option 2: Install via Homebrew (Mac)

```bash
# 1. Uninstall old CLI
pip uninstall databricks-cli

# 2. Install via Homebrew
brew tap databricks/tap
brew install databricks

# 3. Verify
databricks --version
```

### Option 3: Manual Download

1. **Go to releases page:**
   ```
   https://github.com/databricks/cli/releases
   ```

2. **Download for your OS:**
   - macOS: `databricks_cli_X.X.X_darwin_amd64.zip` (Intel)
   - macOS: `databricks_cli_X.X.X_darwin_arm64.zip` (Apple Silicon)
   - Linux: `databricks_cli_X.X.X_linux_amd64.tar.gz`
   - Windows: `databricks_cli_X.X.X_windows_amd64.zip`

3. **Extract and install:**
   ```bash
   # macOS/Linux
   unzip databricks_cli_*.zip
   sudo mv databricks /usr/local/bin/
   
   # Or add to PATH
   export PATH="$PATH:$HOME/bin"
   mv databricks $HOME/bin/
   ```

---

## 🔑 Configuration (After Installation)

### Configure Authentication

```bash
# Interactive configuration
databricks auth login

# You'll be prompted for:
# 1. Databricks Host: https://your-workspace.cloud.databricks.com
# 2. It will open a browser for authentication
# 3. Follow the browser prompts to authorize
```

### Or Use Token Authentication

```bash
# Create config file manually
mkdir -p ~/.databrickscfg

cat > ~/.databrickscfg << 'EOF'
[DEFAULT]
host = https://your-workspace.cloud.databricks.com
token = dapi1234567890abcdef
EOF

# Verify configuration
databricks workspace ls /
```

---

## ✅ Verify Installation

```bash
# 1. Check version
databricks --version
# Expected: Databricks CLI v0.2xx.x

# 2. Check bundle command exists
databricks bundle --help
# Should show bundle subcommands:
#   - deploy
#   - run
#   - validate
#   - etc.

# 3. Test authentication
databricks workspace ls /
# Should list workspace folders

# 4. Validate your bundle
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
databricks bundle validate
# Should validate your databricks.yml
```

---

## 🆕 Key Differences: Old CLI vs New CLI

| Feature | Old CLI | New CLI |
|---------|---------|---------|
| Command | `databricks workspace` | `databricks workspace` |
| Jobs | `databricks jobs` | `databricks jobs` |
| Bundles | ❌ Not supported | ✅ `databricks bundle` |
| Auth | Token only | Token + OAuth |
| Config | `~/.databrickscfg` | `~/.databrickscfg` |
| Python | Python package | Standalone binary |

---

## 📝 Updated Quick Start

Once you've installed the new CLI:

### 1. Configure Authentication
```bash
databricks auth login
# Or use token: databricks configure --token
```

### 2. Validate Bundle
```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
databricks bundle validate
```

### 3. Deploy to DEV
```bash
databricks bundle deploy --target dev
```

### 4. Run Training Job
```bash
databricks bundle run model_training_job --target dev
```

### 5. Check Status
```bash
databricks jobs list | grep "Claims Fraud"
```

---

## 🐛 Troubleshooting

### Issue: "command not found: databricks"

**Solution:**
```bash
# Check if installed
which databricks

# If not found, add to PATH
export PATH="$PATH:/usr/local/bin"

# Or reinstall
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
```

### Issue: "Authentication failed"

**Solution:**
```bash
# Reconfigure
databricks auth login

# Or check your token
cat ~/.databrickscfg

# Test connection
databricks workspace ls /
```

### Issue: "Bundle validation failed"

**Solution:**
```bash
# Check YAML syntax
yamllint databricks.yml

# Run our validation script
python3 validate_bundle.py

# Check error details
databricks bundle validate --verbose
```

---

## 📚 Resources

- **New CLI Documentation:** https://docs.databricks.com/dev-tools/cli/index.html
- **Migration Guide:** https://docs.databricks.com/dev-tools/cli/migrate.html
- **Bundle Documentation:** https://docs.databricks.com/dev-tools/bundles/index.html
- **CLI Releases:** https://github.com/databricks/cli/releases

---

## ✅ Verification Checklist

After installing new CLI, verify:

- [ ] `databricks --version` shows v0.2xx.x or higher
- [ ] `databricks bundle --help` works
- [ ] `databricks auth login` succeeds
- [ ] `databricks workspace ls /` lists folders
- [ ] `databricks bundle validate` passes
- [ ] Ready to deploy!

---

## 🚀 You're Ready!

Once the new CLI is installed:

```bash
# Navigate to project
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Validate
databricks bundle validate

# Deploy
databricks bundle deploy --target dev
```

**Next Step:** Run the installation commands above, then come back to the main deployment guide! 🎉
