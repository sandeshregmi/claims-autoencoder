#!/usr/bin/env bash
# Databricks Bundle Setup Script
# Automates the setup and validation of Databricks Asset Bundle

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_header "Databricks Bundle Setup"

# Step 1: Check Python
print_info "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python installed: $PYTHON_VERSION"
else
    print_error "Python 3 not found"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

# Step 2: Check pip
print_info "Checking pip..."
if command_exists pip; then
    PIP_VERSION=$(pip --version)
    print_success "pip installed: $PIP_VERSION"
else
    print_error "pip not found"
    exit 1
fi

# Step 3: Install Databricks CLI
print_header "Installing Databricks CLI"
if command_exists databricks; then
    print_success "Databricks CLI already installed"
    databricks --version
else
    print_info "Installing Databricks CLI..."
    pip install databricks-cli
    
    if [ $? -eq 0 ]; then
        print_success "Databricks CLI installed"
    else
        print_error "Failed to install Databricks CLI"
        exit 1
    fi
fi

# Step 4: Check Databricks configuration
print_header "Checking Databricks Configuration"
if [ -f ~/.databrickscfg ]; then
    print_success "Databricks configuration found at ~/.databrickscfg"
    
    # Check if host and token are configured
    if grep -q "host" ~/.databrickscfg && grep -q "token" ~/.databrickscfg; then
        print_success "Databricks host and token configured"
    else
        print_warning "Configuration file exists but may be incomplete"
        print_info "Run: databricks configure --token"
    fi
else
    print_warning "Databricks CLI not configured"
    print_info "Please run: databricks configure --token"
    print_info "You'll need:"
    print_info "  - Workspace URL: https://your-workspace.cloud.databricks.com"
    print_info "  - Access token: (generate in workspace settings)"
fi

# Step 5: Validate bundle files
print_header "Validating Bundle Files"

FILES=(
    "databricks.yml:Main bundle configuration"
    "resources/jobs/training_job.yml:Training job configuration"
    "resources/jobs/scoring_job.yml:Scoring job configuration"
    "resources/jobs/monitoring_job.yml:Monitoring job configuration"
    "src/fairness_analysis.py:Fairness analysis module"
    "src/psi_monitoring.py:PSI monitoring module"
    "src/coordination_hooks.py:Coordination hooks"
)

MISSING_FILES=0
for file_info in "${FILES[@]}"; do
    IFS=':' read -ra PARTS <<< "$file_info"
    FILE="${PARTS[0]}"
    DESC="${PARTS[1]}"
    
    if [ -f "$FILE" ]; then
        print_success "$DESC: $FILE"
    else
        print_error "$DESC: $FILE - MISSING"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    print_warning "$MISSING_FILES required files are missing"
fi

# Step 6: Install Python dependencies
print_header "Installing Python Dependencies"
if [ -f "requirements.txt" ]; then
    print_info "Installing from requirements.txt..."
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        print_success "Dependencies installed"
    else
        print_warning "Some dependencies may have failed to install"
    fi
else
    print_warning "requirements.txt not found"
fi

# Step 7: Run validation script
print_header "Running Bundle Validation"
if [ -f "validate_bundle.py" ]; then
    python3 validate_bundle.py
    
    if [ $? -eq 0 ]; then
        print_success "Bundle validation passed"
    else
        print_warning "Bundle validation found issues"
    fi
else
    print_warning "validate_bundle.py not found"
fi

# Step 8: Validate Databricks bundle (if configured)
if [ -f ~/.databrickscfg ]; then
    print_header "Validating Databricks Bundle Configuration"
    
    if databricks bundle validate 2>/dev/null; then
        print_success "Databricks bundle validation passed"
    else
        print_warning "Databricks bundle validation failed or not configured"
        print_info "This is normal if you haven't configured your workspace URL yet"
    fi
fi

# Step 9: Summary
print_header "Setup Summary"

echo "Next steps:"
echo ""
echo "1. Configure Databricks CLI (if not done):"
echo "   ${BLUE}databricks configure --token${NC}"
echo ""
echo "2. Update databricks.yml with your workspace URL"
echo "   Edit line 188 in databricks.yml"
echo ""
echo "3. Validate bundle:"
echo "   ${BLUE}databricks bundle validate --target dev${NC}"
echo ""
echo "4. Deploy to DEV:"
echo "   ${BLUE}databricks bundle deploy --target dev${NC}"
echo ""
echo "5. Run training job:"
echo "   ${BLUE}databricks bundle run model_training_job --target dev${NC}"
echo ""

print_success "Setup script complete!"
print_info "For detailed instructions, see DATABRICKS_SETUP.md"
