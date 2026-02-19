.PHONY: help install install-dev build test test-cov clean lint format serve deploy validate requirements backup

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

PROJECT_NAME := claims-fraud
PYTHON := python3
PIP := pip3

# Help target
help:
	@echo "$(CYAN)=== $(PROJECT_NAME) - Production Makefile ===$(NC)"
	@echo ""
	@echo "$(GREEN)Setup & Installation$(NC)"
	@echo "  make install          Install package with minimal dependencies"
	@echo "  make install-dev      Install with dev dependencies (testing, linting)"
	@echo "  make requirements     Sync requirements.txt"
	@echo ""
	@echo "$(GREEN)Development$(NC)"
	@echo "  make serve            Start Streamlit dashboard (http://localhost:8501)"
	@echo "  make test             Run all tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make lint             Check code quality (pylint, flake8)"
	@echo "  make format           Auto-format code (black, isort)"
	@echo ""
	@echo "$(GREEN)Build & Deployment$(NC)"
	@echo "  make build            Build package wheel (dist/*.whl)"
	@echo "  make validate         Validate Databricks bundle configuration"
	@echo "  make deploy-dev       Deploy to Databricks DEV environment"
	@echo "  make deploy-prod      Deploy to Databricks PROD environment"
	@echo ""
	@echo "$(GREEN)Maintenance$(NC)"
	@echo "  make clean            Remove cache, build artifacts, backups"
	@echo "  make backup           Create timestamped backup of project"
	@echo "  make backup-old       Archive and remove old files (100+ files)"
	@echo ""
	@echo "$(GREEN)Utility$(NC)"
	@echo "  make cache-clear      Clear Python cache (__pycache__, .pytest_cache)"
	@echo "  make env-info         Show environment information"
	@echo ""

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

install:
	@echo "$(GREEN)Installing $(PROJECT_NAME) package...$(NC)"
	$(PIP) install -e . --break-system-packages --no-deps
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev: install
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install pytest pytest-cov black flake8 pylint isort --break-system-packages
	@echo "$(GREEN)✓ Dev dependencies installed$(NC)"

requirements:
	@echo "$(GREEN)Updating requirements.txt from environment...$(NC)"
	$(PIP) freeze > requirements.txt
	@echo "$(GREEN)✓ Requirements updated$(NC)"

# ============================================================================
# DEVELOPMENT
# ============================================================================

serve:
	@echo "$(GREEN)Starting Streamlit dashboard...$(NC)"
	@echo "$(CYAN)Access at: http://localhost:8501$(NC)"
	@echo "$(CYAN)Press Ctrl+C to stop$(NC)"
	@echo ""
	streamlit run src/claims_fraud/ui/app.py

serve-cli:
	@echo "$(GREEN)Starting via CLI entry point...$(NC)"
	claims-fraud serve

test:
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v --tb=short

test-cov:
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=src/claims_fraud --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report generated (htmlcov/index.html)$(NC)"

lint:
	@echo "$(GREEN)Linting code...$(NC)"
	@echo "$(CYAN)Running flake8...$(NC)"
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	@echo "$(CYAN)Running pylint...$(NC)"
	pylint src/claims_fraud --disable=C0111,R0903,R0913 || true
	@echo "$(GREEN)✓ Lint check complete$(NC)"

format:
	@echo "$(GREEN)Formatting code...$(NC)"
	@echo "$(CYAN)Running black...$(NC)"
	black src/ tests/ --line-length=100
	@echo "$(CYAN)Running isort...$(NC)"
	isort src/ tests/ --profile=black
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check:
	@echo "$(GREEN)Checking code format (no changes)...$(NC)"
	black src/ tests/ --line-length=100 --check
	isort src/ tests/ --profile=black --check-only
	@echo "$(GREEN)✓ Format check complete$(NC)"

# ============================================================================
# BUILD & DEPLOYMENT
# ============================================================================

build: clean
	@echo "$(GREEN)Building package...$(NC)"
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build
	@echo "$(GREEN)✓ Package built$(NC)"
	@ls -lh dist/

validate:
	@echo "$(GREEN)Validating Databricks bundle...$(NC)"
	@if command -v databricks &> /dev/null; then \
		databricks bundle validate; \
		echo "$(GREEN)✓ Bundle validation complete$(NC)"; \
	else \
		echo "$(RED)✗ Databricks CLI not installed. Install with: pip install databricks-cli$(NC)"; \
		exit 1; \
	fi

deploy-dev: build validate
	@echo "$(GREEN)Deploying to Databricks DEV...$(NC)"
	@if command -v databricks &> /dev/null; then \
		databricks bundle deploy --target dev; \
		echo "$(GREEN)✓ DEV deployment complete$(NC)"; \
	else \
		echo "$(RED)✗ Databricks CLI not installed$(NC)"; \
		exit 1; \
	fi

deploy-prod: build validate
	@echo "$(YELLOW)⚠️  Deploying to PRODUCTION...$(NC)"
	@read -p "Are you sure? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		if command -v databricks &> /dev/null; then \
			databricks bundle deploy --target prod; \
			echo "$(GREEN)✓ PROD deployment complete$(NC)"; \
		else \
			echo "$(RED)✗ Databricks CLI not installed$(NC)"; \
			exit 1; \
		fi; \
	else \
		echo "$(YELLOW)Deployment cancelled$(NC)"; \
	fi

# ============================================================================
# MAINTENANCE
# ============================================================================

clean: cache-clear
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Clean complete$(NC)"

cache-clear:
	@echo "$(GREEN)Clearing cache...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "catboost_info" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cache cleared$(NC)"

backup:
	@echo "$(GREEN)Creating project backup...$(NC)"
	@BACKUP_DIR="backup_$$(date +%Y%m%d_%H%M%S)" && \
	mkdir -p $$BACKUP_DIR && \
	tar -czf $$BACKUP_DIR.tar.gz \
		--exclude='.git' \
		--exclude='venv' \
		--exclude='__pycache__' \
		--exclude='.pytest_cache' \
		--exclude='dist' \
		--exclude='build' \
		--exclude='*.egg-info' \
		. && \
	echo "$(GREEN)✓ Backup created: $$BACKUP_DIR.tar.gz$(NC)"

backup-old:
	@echo "$(YELLOW)Creating backup of old files (100+ files to archive)...$(NC)"
	@BACKUP_DIR="backup_old_$$(date +%Y%m%d_%H%M%S)" && \
	mkdir -p $$BACKUP_DIR && \
	echo "$(CYAN)Moving old documentation and scripts...$(NC)" && \
	find . -maxdepth 1 \( -name "*_COMPLETE.md" -o -name "*_FIX.md" -o -name "*_GUIDE.md" -o -name "add_*.py" -o -name "apply_*.py" -o -name "fix_*.sh" \) -exec mv {} $$BACKUP_DIR/ \; 2>/dev/null || true && \
	echo "$(GREEN)✓ Old files archived to: $$BACKUP_DIR$(NC)" && \
	du -sh $$BACKUP_DIR

# ============================================================================
# ENVIRONMENT & INFO
# ============================================================================

env-info:
	@echo "$(GREEN)=== Environment Information ===$(NC)"
	@echo "$(CYAN)Python:$(NC)"
	@$(PYTHON) --version
	@echo ""
	@echo "$(CYAN)Pip:$(NC)"
	@$(PIP) --version
	@echo ""
	@echo "$(CYAN)Project structure:$(NC)"
	@echo "  Source: src/claims_fraud/"
	@echo "  Tests:  tests/"
	@echo "  Data:   data/"
	@echo "  Models: models/"
	@echo "  Config: config/"

# ============================================================================
# QUICK COMMANDS
# ============================================================================

install-and-serve: install
	@echo "$(GREEN)Starting dashboard...$(NC)"
	streamlit run src/claims_fraud/ui/app.py

install-dev-and-test: install-dev
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v

# ============================================================================
# DATABRICKS SPECIFIC
# ============================================================================

dbc-init:
	@echo "$(GREEN)Initializing Databricks CLI...$(NC)"
	@if command -v databricks &> /dev/null; then \
		databricks configure --token; \
		echo "$(GREEN)✓ Databricks configured$(NC)"; \
	else \
		echo "$(RED)✗ Installing Databricks CLI...$(NC)"; \
		$(PIP) install databricks-cli; \
		databricks configure --token; \
	fi

dbc-status:
	@echo "$(GREEN)Checking Databricks connection...$(NC)"
	@if command -v databricks &> /dev/null; then \
		databricks workspace list; \
	else \
		echo "$(RED)✗ Databricks CLI not installed$(NC)"; \
	fi

# ============================================================================
# DEFAULT TARGET
# ============================================================================

.DEFAULT_GOAL := help
