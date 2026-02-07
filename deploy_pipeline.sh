#!/bin/bash
# Deploy the complete ML pipeline to Databricks

echo "=========================================="
echo "Deploying Complete ML Pipeline"
echo "=========================================="
echo ""

cd /Users/sregmi/pytorch-tabular-mcp/claims-autoencoder

echo "📦 Deploying 3 jobs to Databricks:"
echo "  1. Training Pipeline (with parallel execution)"
echo "  2. Monitoring Job (every 4 hours)"
echo "  3. Batch Scoring Job (every 6 hours)"
echo ""

databricks bundle deploy --target dev

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ DEPLOYMENT SUCCESSFUL!"
    echo ""
    echo "📋 Deployed Jobs:"
    databricks jobs list | grep "Claims Fraud"
    echo ""
    echo "🚀 Next Steps:"
    echo ""
    echo "1. Test training pipeline:"
    echo "   databricks bundle run model_training_job --target dev"
    echo ""
    echo "2. View in Databricks UI:"
    echo "   https://dbc-d4506e69-bbc8.cloud.databricks.com/#job/list"
    echo ""
    echo "3. Enable schedules (currently PAUSED):"
    echo "   - Edit databricks.yml: change pause_status to UNPAUSED"
    echo "   - Redeploy: databricks bundle deploy --target dev"
    echo ""
else
    echo ""
    echo "❌ Deployment failed"
    echo "Check the errors above"
fi
