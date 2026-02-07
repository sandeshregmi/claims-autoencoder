# 🚨 IMPORTANT: CLI Installation Required

## Issue You're Experiencing

```
Error: No such command 'bundle'.
Warning: The version of the CLI you are using is deprecated.
```

You have the **old Databricks CLI** installed. Asset Bundles require the **new CLI**.

---

## ✅ Quick Fix (5 minutes)

### Step 1: Uninstall Old CLI
```bash
pip uninstall databricks-cli
```

### Step 2: Install New CLI

**Option A - Quick Install (Recommended):**
```bash
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
```

**Option B - Homebrew (Mac):**
```bash
brew tap databricks/tap
brew install databricks
```

### Step 3: Verify Installation
```bash
databricks --version
# Should show: Databricks CLI v0.2xx.x or higher

databricks bundle --help
# Should show bundle commands
```

### Step 4: Configure Authentication
```bash
databricks auth login
# This will open a browser for OAuth authentication

# OR use token:
databricks configure --token
# You'll need:
#   Host: https://your-workspace.cloud.databricks.com
#   Token: (generate in workspace → Settings → Developer → Access tokens)
```

### Step 5: Test
```bash
# Test connection
databricks workspace ls /

# Validate bundle
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder
databricks bundle validate
```

---

## 📝 Automated Installation Script

We've created an automated script for you:

```bash
cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

# Make executable
chmod +x install_databricks_cli.sh

# Run installation
./install_databricks_cli.sh
```

This script will:
1. ✅ Detect and uninstall old CLI
2. ✅ Install new CLI
3. ✅ Verify installation
4. ✅ Check configuration
5. ✅ Validate your bundle

---

## 🔍 What Changed?

| Feature | Old CLI | New CLI |
|---------|---------|---------|
| Package | `databricks-cli` (pip) | Standalone binary |
| Bundles | ❌ Not supported | ✅ `databricks bundle` |
| Auth | Token only | Token + OAuth |
| Commands | Same | Same + bundle commands |

---

## ✅ After Installation

Once installed, continue with deployment:

```bash
# 1. Validate
databricks bundle validate --target dev

# 2. Deploy
databricks bundle deploy --target dev

# 3. Run training
databricks bundle run model_training_job --target dev
```

---

## 📚 Detailed Guides

- **CLI_MIGRATION_GUIDE.md** - Complete migration instructions
- **DATABRICKS_SETUP.md** - Full setup guide
- **DATABRICKS_QUICK_REFERENCE.md** - Command reference

---

## 🆘 Troubleshooting

### "command not found: databricks"

Add to PATH:
```bash
export PATH="$PATH:/usr/local/bin"

# Or restart your terminal
```

### "Authentication failed"

Reconfigure:
```bash
databricks auth login
```

### Still having issues?

Run the automated installer:
```bash
./install_databricks_cli.sh
```

---

## 🚀 Next Steps

1. **Install new CLI** (commands above)
2. **Configure authentication**
3. **Validate bundle**
4. **Deploy to DEV**
5. **Run your first training job!**

**Estimated time:** 10 minutes including installation 🎉
