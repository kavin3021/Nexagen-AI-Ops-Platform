# NexaGen AI Ops Platform - Makefile
# Automation commands for development, deployment, and operations

.PHONY: help setup install-deps configure-azure configure-gcp \
        deploy-azure deploy-gcp deploy-all destroy-azure destroy-gcp destroy-all \
        test test-unit test-integration test-load \
        lint format security-scan \
        data-prepare data-validate \
        demo health-check \
        monitor logs cost-report \
        clean

# Default target
.DEFAULT_GOAL := help

# Color codes for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Project variables
PROJECT_NAME := nexagen
VERSION := 1.0.0
PYTHON := python3
PIP := pip3
TERRAFORM_AZURE := terraform/azure
TERRAFORM_GCP := terraform/gcp

help: ## Show this help message
	@echo "$(GREEN)NexaGen AI Ops Platform - Available Commands$(NC)"
	@echo "=================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ===========================================
# SETUP & INSTALLATION
# ===========================================

setup: ## Complete project setup (install deps, configure tools)
	@echo "$(GREEN)Setting up NexaGen AI Ops Platform...$(NC)"
	make install-deps
	make configure-tools
	make prepare-env
	@echo "$(GREEN)✅ Setup complete!$(NC)"

install-deps: ## Install Python dependencies
	@echo "$(YELLOW)Installing Python dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

configure-tools: ## Configure development tools
	@echo "$(YELLOW)Configuring development tools...$(NC)"
	pre-commit install
	@echo "$(GREEN)✅ Tools configured$(NC)"

prepare-env: ## Prepare environment files
	@echo "$(YELLOW)Preparing environment files...$(NC)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)⚠️  Please update .env with your credentials$(NC)"; \
	fi
	@echo "$(GREEN)✅ Environment prepared$(NC)"

# ===========================================
# CLOUD CONFIGURATION
# ===========================================

configure-azure: ## Configure Azure CLI and authentication
	@echo "$(YELLOW)Configuring Azure CLI...$(NC)"
	az login
	az account show
	@echo "$(GREEN)✅ Azure configured$(NC)"

configure-gcp: ## Configure GCP CLI and authentication
	@echo "$(YELLOW)Configuring GCP CLI...$(NC)"
	gcloud auth login
	gcloud auth application-default login
	gcloud config list
	@echo "$(GREEN)✅ GCP configured$(NC)"

# ===========================================
# INFRASTRUCTURE DEPLOYMENT
# ===========================================

deploy-azure: ## Deploy Azure infrastructure
	@echo "$(YELLOW)Deploying Azure infrastructure...$(NC)"
	cd $(TERRAFORM_AZURE) && \
	terraform init && \
	terraform plan && \
	terraform apply -auto-approve
	@echo "$(GREEN)✅ Azure infrastructure deployed$(NC)"

deploy-gcp: ## Deploy GCP infrastructure
	@echo "$(YELLOW)Deploying GCP infrastructure...$(NC)"
	cd $(TERRAFORM_GCP) && \
	terraform init && \
	terraform plan && \
	terraform apply -auto-approve
	@echo "$(GREEN)✅ GCP infrastructure deployed$(NC)"

deploy-all: ## Deploy infrastructure on both clouds
	@echo "$(GREEN)Deploying multi-cloud infrastructure...$(NC)"
	make deploy-azure
	make deploy-gcp
	@echo "$(GREEN)✅ Multi-cloud deployment complete$(NC)"

# Demo deployment (minimal resources for recording)
deploy-demo: ## Deploy minimal resources for demo
	@echo "$(YELLOW)Deploying demo resources (minimal cost)...$(NC)"
	cd $(TERRAFORM_AZURE) && \
	terraform apply -auto-approve -var="demo_mode=true"
	cd ../$(TERRAFORM_GCP) && \
	terraform apply -auto-approve -var="demo_mode=true"
	@echo "$(GREEN)✅ Demo resources deployed$(NC)"

# ===========================================
# INFRASTRUCTURE DESTRUCTION
# ===========================================

destroy-azure: ## Destroy Azure infrastructure
	@echo "$(RED)Destroying Azure infrastructure...$(NC)"
	cd $(TERRAFORM_AZURE) && \
	terraform destroy -auto-approve
	@echo "$(RED)💥 Azure infrastructure destroyed$(NC)"

destroy-gcp: ## Destroy GCP infrastructure
	@echo "$(RED)Destroying GCP infrastructure...$(NC)"
	cd $(TERRAFORM_GCP) && \
	terraform destroy -auto-approve
	@echo "$(RED)💥 GCP infrastructure destroyed$(NC)"

destroy-all: ## Destroy infrastructure on both clouds
	@echo "$(RED)Destroying all infrastructure...$(NC)"
	make destroy-azure
	make destroy-gcp
	@echo "$(RED)💥 All infrastructure destroyed$(NC)"

# ===========================================
# APPLICATION DEPLOYMENT
# ===========================================

run-local: ## Run application locally
	@echo "$(YELLOW)Starting local application...$(NC)"
	docker-compose up -d
	uvicorn llm-applications.main:app --reload --host 0.0.0.0 --port 8000
	@echo "$(GREEN)✅ Application running on http://localhost:8000$(NC)"

stop-local: ## Stop local application
	@echo "$(YELLOW)Stopping local application...$(NC)"
	docker-compose down
	@echo "$(GREEN)✅ Local application stopped$(NC)"

# ===========================================
# DATA PREPARATION
# ===========================================

data-prepare: ## Prepare training and evaluation data
	@echo "$(YELLOW)Preparing training data...$(NC)"
	$(PYTHON) scripts/data_processing/prepare_training_data.py
	$(PYTHON) scripts/data_processing/generate_embeddings.py
	@echo "$(GREEN)✅ Training data prepared$(NC)"

data-validate: ## Validate data quality and format
	@echo "$(YELLOW)Validating data quality...$(NC)"
	$(PYTHON) scripts/data_processing/data_validation.py
	@echo "$(GREEN)✅ Data validation complete$(NC)"

# ===========================================
# MODEL OPERATIONS
# ===========================================

finetune-azure: ## Run fine-tuning on Azure ML
	@echo "$(YELLOW)Starting Azure ML fine-tuning...$(NC)"
	$(PYTHON) llm-applications/azure-openai/finetune_pipeline.py
	@echo "$(GREEN)✅ Azure fine-tuning complete$(NC)"

finetune-gcp: ## Run fine-tuning on GCP Vertex AI
	@echo "$(YELLOW)Starting Vertex AI fine-tuning...$(NC)"
	$(PYTHON) llm-applications/vertex-ai/finetune_pipeline.py
	@echo "$(GREEN)✅ GCP fine-tuning complete$(NC)"

evaluate-models: ## Run model evaluation pipeline
	@echo "$(YELLOW)Running model evaluation...$(NC)"
	$(PYTHON) llm-applications/vertex-ai/model_evaluation.py
	@echo "$(GREEN)✅ Model evaluation complete$(NC)"

# ===========================================
# TESTING
# ===========================================

test: ## Run all tests
	@echo "$(YELLOW)Running test suite...$(NC)"
	make test-unit
	make test-integration
	@echo "$(GREEN)✅ All tests passed$(NC)"

test-unit: ## Run unit tests
	@echo "$(YELLOW)Running unit tests...$(NC)"
	pytest tests/unit/ -v --cov=. --cov-report=html
	@echo "$(GREEN)✅ Unit tests passed$(NC)"

test-integration: ## Run integration tests
	@echo "$(YELLOW)Running integration tests...$(NC)"
	pytest tests/integration/ -v
	@echo "$(GREEN)✅ Integration tests passed$(NC)"

test-load: ## Run load tests
	@echo "$(YELLOW)Running load tests...$(NC)"
	locust -f tests/performance/load_test.py --host=http://localhost:8000
	@echo "$(GREEN)✅ Load tests complete$(NC)"

# ===========================================
# CODE QUALITY
# ===========================================

lint: ## Run code linting
	@echo "$(YELLOW)Running linting...$(NC)"
	flake8 llm-applications/ multimodal/ scripts/
	mypy llm-applications/ multimodal/
	pylint llm-applications/ multimodal/ scripts/
	@echo "$(GREEN)✅ Linting complete$(NC)"

format: ## Format code
	@echo "$(YELLOW)Formatting code...$(NC)"
	black llm-applications/ multimodal/ scripts/ tests/
	isort llm-applications/ multimodal/ scripts/ tests/
	@echo "$(GREEN)✅ Code formatted$(NC)"

security-scan: ## Run security scans
	@echo "$(YELLOW)Running security scans...$(NC)"
	bandit -r llm-applications/ multimodal/ scripts/
	safety check
	@echo "$(GREEN)✅ Security scan complete$(NC)"

# ===========================================
# DEMO & PRESENTATION
# ===========================================

demo: ## Run complete demo pipeline
	@echo "$(GREEN)Starting NexaGen AI Ops Platform Demo...$(NC)"
	@echo "$(YELLOW)1. Deploying demo resources...$(NC)"
	make deploy-demo
	@echo "$(YELLOW)2. Running health checks...$(NC)"
	make health-check
	@echo "$(YELLOW)3. Testing multimodal pipeline...$(NC)"
	$(PYTHON) multimodal/demo.py
	@echo "$(YELLOW)4. Testing fine-tuned models...$(NC)"
	$(PYTHON) llm-applications/vertex-ai/demo_inference.py
	@echo "$(YELLOW)5. Showing monitoring dashboards...$(NC)"
	make monitor
	@echo "$(YELLOW)6. Generating cost report...$(NC)"
	make cost-report
	@echo "$(GREEN)✅ Demo complete! Don't forget to run 'make destroy-all'$(NC)"

health-check: ## Run system health checks
	@echo "$(YELLOW)Running health checks...$(NC)"
	$(PYTHON) scripts/deployment/health_check.py
	@echo "$(GREEN)✅ Health check complete$(NC)"

# ===========================================
# MONITORING & OBSERVABILITY
# ===========================================

monitor: ## Open monitoring dashboards
	@echo "$(YELLOW)Opening monitoring dashboards...$(NC)"
	@echo "$(BLUE)Azure Application Insights:$(NC) https://portal.azure.com/"
	@echo "$(BLUE)GCP Cloud Monitoring:$(NC) https://console.cloud.google.com/monitoring"
	@echo "$(BLUE)Grafana (local):$(NC) http://localhost:3000"
	@echo "$(GREEN)✅ Dashboard links displayed$(NC)"

logs: ## View application logs
	@echo "$(YELLOW)Viewing application logs...$(NC)"
	docker-compose logs -f

cost-report: ## Generate cost report
	@echo "$(YELLOW)Generating cost report...$(NC)"
	$(PYTHON) scripts/utilities/cost_calculator.py
	@echo "$(GREEN)✅ Cost report generated$(NC)"

# ===========================================
# MAINTENANCE & CLEANUP
# ===========================================

clean: ## Clean up temporary files and caches
	@echo "$(YELLOW)Cleaning up temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

reset: ## Reset project to initial state (DANGEROUS)
	@echo "$(RED)⚠️  This will destroy ALL resources and data!$(NC)"
	@read -p "Type 'RESET' to confirm: " confirm && [ "$$confirm" = "RESET" ] || exit 1
	make destroy-all
	make clean
	rm -f .env
	@echo "$(RED)💥 Project reset complete$(NC)"

# ===========================================
# DOCUMENTATION
# ===========================================

docs: ## Generate documentation
	@echo "$(YELLOW)Generating documentation...$(NC)"
	mkdocs build
	@echo "$(GREEN)✅ Documentation generated$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(YELLOW)Serving documentation...$(NC)"
	mkdocs serve
	@echo "$(GREEN)✅ Documentation available at http://localhost:8000$(NC)"

# ===========================================
# CI/CD HELPERS
# ===========================================

ci-test: ## Run CI test pipeline
	@echo "$(YELLOW)Running CI test pipeline...$(NC)"
	make install-deps
	make lint
	make test-unit
	make security-scan
	@echo "$(GREEN)✅ CI tests passed$(NC)"

ci-deploy: ## Run CI deployment pipeline
	@echo "$(YELLOW)Running CI deployment pipeline...$(NC)"
	make configure-azure
	make configure-gcp
	make deploy-all
	make health-check
	@echo "$(GREEN)✅ CI deployment complete$(NC)"

# ===========================================
# DEVELOPMENT HELPERS
# ===========================================

dev-setup: ## Setup development environment
	@echo "$(YELLOW)Setting up development environment...$(NC)"
	make setup
	make run-local
	@echo "$(GREEN)✅ Development environment ready$(NC)"

jupyter: ## Start Jupyter Lab for development
	@echo "$(YELLOW)Starting Jupyter Lab...$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
	@echo "$(GREEN)✅ Jupyter Lab running$(NC)"

# ===========================================
# EMERGENCY PROCEDURES
# ===========================================

emergency-stop: ## Emergency stop all resources (cost-saving)
	@echo "$(RED)⚠️  Emergency stop - destroying all resources!$(NC)"
	make destroy-all
	make stop-local
	@echo "$(RED)💥 Emergency stop complete$(NC)"

budget-alert: ## Check current spending and alert if over budget
	@echo "$(YELLOW)Checking budget status...$(NC)"
	$(PYTHON) scripts/utilities/cost_calculator.py --alert
	@echo "$(GREEN)✅ Budget check complete$(NC)"

# ===========================================
# UTILITIES
# ===========================================

version: ## Show project version
	@echo "$(GREEN)NexaGen AI Ops Platform v$(VERSION)$(NC)"

status: ## Show project status
	@echo "$(GREEN)NexaGen AI Ops Platform Status$(NC)"
	@echo "================================"
	@echo "$(BLUE)Version:$(NC) $(VERSION)"
	@echo "$(BLUE)Python:$(NC) $(shell $(PYTHON) --version)"
	@echo "$(BLUE)Docker:$(NC) $(shell docker --version)"
	@echo "$(BLUE)Terraform:$(NC) $(shell terraform --version | head -n1)"
	@echo "$(BLUE)Azure CLI:$(NC) $(shell az --version | head -n1)"
	@echo "$(BLUE)GCP CLI:$(NC) $(shell gcloud --version | head -n1)"

# Check if .env file exists and warn if not
check-env:
	@if [ ! -f .env ]; then \
		echo "$(RED)⚠️  .env file not found! Run 'make prepare-env' first$(NC)"; \
		exit 1; \
	fi

# Dependency check for critical commands
check-deps:
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "$(RED)❌ Python not found$(NC)"; exit 1; }
	@command -v terraform >/dev/null 2>&1 || { echo "$(RED)❌ Terraform not found$(NC)"; exit 1; }
	@command -v az >/dev/null 2>&1 || { echo "$(RED)❌ Azure CLI not found$(NC)"; exit 1; }
	@command -v gcloud >/dev/null 2>&1 || { echo "$(RED)❌ GCP CLI not found$(NC)"; exit 1; }

# Add dependency checks to critical targets
deploy-azure deploy-gcp deploy-all: check-deps check-env
destroy-azure destroy-gcp destroy-all: check-deps check-env