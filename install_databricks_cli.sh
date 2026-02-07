#!/usr/bin/env bash
# Databricks CLI Installation & Setup Script
# Handles migration from old CLI to new CLI with Asset Bundle support

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

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_header "Databricks CLI Setup & Migration"

# Step 1: Check for old CLI
print_info "Checking for existing Databricks CLI..."

OLD_CLI_INSTALLED=false
NEW_CLI_INSTALLED=false

if command_exists databricks; then
    # Check if it's the old CLI
    if databricks bundle --help >/dev/null 2>&1; then
        NEW_CLI_INSTALLED=true
        CLI_VERSION=$(databricks --version 2>&1)
        print_success "New Databricks CLI already installed: $CLI_VERSION"
    else
        OLD_CLI_INSTALLED=true
        print_warning "Old Databricks CLI detected (does not support bundles)"
        print_info "The old CLI will be replaced with the new CLI"
    fi
else
    print_info "No Databricks CLI found. Will install new CLI."
fi

# Step 2: Install New CLI
if [ "$NEW_CLI_INSTALLED" = false ]; then
    print_header "Installing New Databricks CLI"
    
    # Uninstall old CLI if present
    if [ "$OLD_CLI_INSTALLED" = true ]; then
        print_info "Uninstalling old CLI..."
        pip uninstall -y databricks-cli 2>/dev/null || true
        print_success "Old CLI uninstalled"
    fi
    
    # Detect OS
    OS="$(uname -s)"
    case "${OS}" in
        Darwin*)
            print_info "macOS detected"
            
            # Check if Homebrew is available
            if command_exists brew; then
                print_info "Installing via Homebrew..."
                brew tap databricks/tap 2>/dev/null || true
                brew install databricks
                print_success "Databricks CLI installed via Homebrew"
            else
                print_info "Homebrew not found. Installing via curl..."
                curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
                print_success "Databricks CLI installed"
            fi
            ;;
        Linux*)
            print_info "Linux detected"
            print_info "Installing via curl..."
            curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
            print_success "Databricks CLI installed"
            ;;
        *)
            print_error "Unsupported OS: ${OS}"
            print_info "Please install manually from: https://github.com/databricks/cli/releases"
            exit 1
            ;;
    esac
    
    # Verify installation
    if command_exists databricks; then
        CLI_VERSION=$(databricks --version 2>&1)
        print_success "New CLI verified: $CLI_VERSION"
    else
        print_error "Installation failed. CLI not found in PATH."
        print_info "You may need to restart your terminal or run: export PATH=\"\$PATH:/usr/local/bin\""
        exit 1
    fi
fi

# Step 3: Verify bundle command
print_header "Verifying Bundle Support"

if databricks bundle --help >/dev/null 2>&1; then
    print_success "Bundle command available"
else
    print_error "Bundle command not available"
    print_info "Please ensure you have the latest version of the CLI"
    print_info "Download from: https://github.com/databricks/cli/releases"
    exit 1
fi

# Step 4: Check configuration
print_header "Checking Databricks Configuration"

if [ -f ~/.databrickscfg ]; then
    print_success "Configuration file found at ~/.databrickscfg"
    
    # Check if host and token are configured
    if grep -q "host" ~/.databrickscfg && grep -q "token" ~/.databrickscfg; then
        print_success "Host and token configured"
        
        # Test connection
        print_info "Testing connection..."
        if databricks workspace ls / >/dev/null 2>&1; then
            print_success "Connection test passed"
        else
            print_warning "Connection test failed. Please check your credentials."
            print_info "Run: databricks auth login"
        fi
    else
        print_warning "Configuration incomplete"
        print_info "Run: databricks auth login"
    fi
else
    print_warning "No configuration found"
    print_info "Please configure the CLI:"
    echo ""
    print_info "Option 1 - OAuth (Recommended):"
    echo "  ${BLUE}databricks auth login${NC}"
    echo ""
    print_info "Option 2 - Token:"
    echo "  ${BLUE}databricks configure --token${NC}"
    echo "  You'll need:"
    echo "    - Host: https://your-workspace.cloud.databricks.com"
    echo "    - Token: (generate in workspace settings → Developer → Access tokens)"
fi

# Step 5: Validate bundle files
print_header "Validating Project Files"

cd "$(dirname "$0")"

REQUIRED_FILES=(
    "databricks.yml:Main bundle configuration"
    "resources/jobs/training_job.yml:Training job"
    "resources/jobs/scoring_job.yml:Scoring job"
    "resources/jobs/monitoring_job.yml:Monitoring job"
)

MISSING_FILES=0
for file_info in "${REQUIRED_FILES[@]}"; do
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
    print_error "$MISSING_FILES required files are missing"
    exit 1
fi

# Step 6: Run bundle validation
if [ -f ~/.databrickscfg ]; then
    print_header "Validating Databricks Bundle"
    
    print_info "Running: databricks bundle validate"
    
    if databricks bundle validate 2>&1 | tee /tmp/bundle_validation.log; then
        print_success "Bundle validation passed!"
    else
        print_error "Bundle validation failed"
        echo ""
        print_info "Common issues:"
        echo "  1. Update workspace host in databricks.yml (line 188)"
        echo "  2. Check YAML syntax with: yamllint databricks.yml"
        echo "  3. Run: python3 validate_bundle.py for detailed checks"
        echo ""
        cat /tmp/bundle_validation.log
    fi
else
    print_warning "Skipping bundle validation (CLI not configured)"
fi

# Step 7: Summary
print_header "Setup Summary"

if [ "$NEW_CLI_INSTALLED" = true ] || command_exists databricks; then
    print_success "✓ New Databricks CLI installed and verified"
else
    print_error "✗ CLI installation incomplete"
fi

if [ -f ~/.databrickscfg ]; then
    print_success "✓ CLI configured"
else
    print_warning "⚠ CLI not configured yet"
fi

echo ""
echo "Next steps:"
echo ""

if [ ! -f ~/.databrickscfg ]; then
    echo "1. Configure CLI:"
    echo "   ${BLUE}databricks auth login${NC}"
    echo ""
fi

echo "2. Update databricks.yml with your workspace URL"
echo "   Edit line 188: host: https://YOUR-WORKSPACE.cloud.databricks.com"
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
print_info "For detailed instructions, see CLI_MIGRATION_GUIDE.md"
