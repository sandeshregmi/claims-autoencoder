# 🚀 Installation Guide

## Quick Install (One Command)

```bash
chmod +x install.sh && ./install.sh
```

This will:
1. ✅ Clear Python cache
2. ✅ Install the package
3. ✅ Test that everything works

---

## Manual Install (If Preferred)

```bash
# Clear cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Install
pip install -e . --break-system-packages

# Test
python3 test_config.py
```

---

## After Installation

Test that it works:

```bash
python3 -c "from claims_fraud.config.manager import load_config; print('✅ Works!')"
```

Or run the full test:

```bash
python3 test_config.py
```

---

## Expected Output

```
🚀 Installing claims-fraud package...
======================================================================

1. Clearing Python cache...
   ✅ Cache cleared

2. Installing package...
   ✅ Package installed successfully!

3. Testing installation...
   ✅ Package version: 2.0.0
   ✅ load_config available
   ✅ Config loaded: batch_size=256

======================================================================
🎉 Installation complete!
======================================================================
```

---

## Troubleshooting

If you get "externally-managed-environment" error, the script uses `--break-system-packages` flag automatically.

If that doesn't work, create a virtual environment:

```bash
conda create -n claims-fraud python=3.10
conda activate claims-fraud
pip install -e .
```

---

## Verify Installation

```bash
python3 -c "
from claims_fraud import __version__
from claims_fraud.config.manager import load_config
config = load_config('config/config.yaml')
print(f'✅ Version: {__version__}')
print(f'✅ Batch size: {config.training.batch_size}')
print(f'✅ High risk: \${config.business_rules.fraud_thresholds.claim_amount_high_risk:,}')
"
```

---

## Ready to Use!

After installation, you can import anywhere:

```python
from claims_fraud.config.manager import load_config
from claims_fraud.data.validation import DataValidator
from claims_fraud.core.business_rules import BusinessRulesEngine

config = load_config('config/config.yaml')
# Use the config...
```

🎉 Done!
