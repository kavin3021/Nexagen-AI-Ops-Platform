# NexaGen AI Ops Platform - Azure Infrastructure
# Terraform configuration for Azure AI Foundry and supporting services

terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~>3.0"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~>2.0"
    }
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# Data source for current client configuration
data "azurerm_client_config" "current" {}

# Random string for unique resource naming
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = var.demo_mode ? "rg-${var.project_name}-demo" : "rg-${var.project_name}-${var.environment}"
  location = var.location

  tags = local.common_tags
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "vnet-${var.project_name}-${var.environment}"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = local.common_tags
}

# Subnet for AI services
resource "azurerm_subnet" "ai_services" {
  name                 = "subnet-ai-services"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.1.0/24"]

  service_endpoints = [
    "Microsoft.Storage",
    "Microsoft.KeyVault",
    "Microsoft.CognitiveServices"
  ]
}

# Subnet for compute resources
resource "azurerm_subnet" "compute" {
  name                 = "subnet-compute"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.2.0/24"]
}

# Network Security Group
resource "azurerm_network_security_group" "main" {
  name                = "nsg-${var.project_name}-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "AllowHTTPS"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowHTTP"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = local.common_tags
}

# Associate NSG with AI services subnet
resource "azurerm_subnet_network_security_group_association" "ai_services" {
  subnet_id                 = azurerm_subnet.ai_services.id
  network_security_group_id = azurerm_network_security_group.main.id
}

# Storage Account for AI Foundry
resource "azurerm_storage_account" "ai_foundry" {
  name                     = "st${var.project_name}ai${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = var.demo_mode ? "Standard" : "Standard"
  account_replication_type = var.demo_mode ? "LRS" : "ZRS"
  
  # Enable hierarchical namespace for Data Lake Storage Gen2
  is_hns_enabled = true
  
  # Security configurations
  public_network_access_enabled = true
  min_tls_version               = "TLS1_2"
  
  blob_properties {
    versioning_enabled       = true
    change_feed_enabled     = true
    delete_retention_policy {
      days = var.demo_mode ? 1 : 7
    }
    container_delete_retention_policy {
      days = var.demo_mode ? 1 : 7
    }
  }

  tags = local.common_tags
}

# Storage Container for documents
resource "azurerm_storage_container" "documents" {
  name                  = "documents"
  storage_account_name  = azurerm_storage_account.ai_foundry.name
  container_access_type = "private"
}

# Storage Container for models
resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.ai_foundry.name
  container_access_type = "private"
}

# Key Vault for secrets management
resource "azurerm_key_vault" "main" {
  name                       = "kv-${var.project_name}-${random_string.suffix.result}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  
  soft_delete_retention_days = var.demo_mode ? 7 : 30
  purge_protection_enabled   = false
  
  public_network_access_enabled = true
  
  network_acls {
    default_action = "Allow"
    bypass         = "AzureServices"
  }

  # Access policy for current user
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    key_permissions = [
      "Get", "List", "Update", "Create", "Import", "Delete", "Recover",
      "Backup", "Restore", "Decrypt", "Encrypt", "UnwrapKey", "WrapKey",
      "Verify", "Sign", "Purge"
    ]

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Recover", "Backup", "Restore", "Purge"
    ]

    certificate_permissions = [
      "Get", "List", "Update", "Create", "Import", "Delete", "Recover",
      "Backup", "Restore", "ManageContacts", "ManageIssuers", "GetIssuers",
      "ListIssuers", "SetIssuers", "DeleteIssuers", "Purge"
    ]
  }

  tags = local.common_tags
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "law-${var.project_name}-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = var.demo_mode ? 30 : 90

  tags = local.common_tags
}

# Application Insights
resource "azurerm_application_insights" "main" {
  name                = "appi-${var.project_name}-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  retention_in_days   = var.demo_mode ? 30 : 90

  tags = local.common_tags
}

# Azure OpenAI Service
resource "azurerm_cognitive_account" "openai" {
  name                = "cog-${var.project_name}-openai-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "OpenAI"
  sku_name            = "S0"

  public_network_access_enabled = true
  custom_subdomain_name         = "nexagen-openai-${random_string.suffix.result}"

  network_acls {
    default_action = "Allow"
  }

  tags = local.common_tags
}

# Azure OpenAI Deployment - GPT-4o-mini
resource "azurerm_cognitive_deployment" "gpt4o_mini" {
  name                 = var.gpt_deployment_name
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = "gpt-4o-mini"
    version = "2024-07-18"
  }

  scale {
    type     = "Standard"
    capacity = var.demo_mode ? 1 : 10
  }
}

# Azure OpenAI Deployment - Text Embedding
resource "azurerm_cognitive_deployment" "text_embedding" {
  name                 = var.embedding_deployment_name
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = "text-embedding-ada-002"
    version = "2"
  }

  scale {
    type     = "Standard"
    capacity = var.demo_mode ? 1 : 50
  }
}

# Azure AI Search Service
resource "azurerm_search_service" "main" {
  name                = "srch-${var.project_name}-${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.demo_mode ? "free" : "basic"
  replica_count       = 1
  partition_count     = 1

  public_network_access_enabled = true

  tags = local.common_tags
}

# Computer Vision Service
resource "azurerm_cognitive_account" "computer_vision" {
  name                = "cog-${var.project_name}-cv-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "ComputerVision"
  sku_name            = "F0"  # Free tier

  public_network_access_enabled = true

  tags = local.common_tags
}

# Content Safety Service
resource "azurerm_cognitive_account" "content_safety" {
  name                = "cog-${var.project_name}-safety-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "ContentSafety"
  sku_name            = "F0"  # Free tier

  public_network_access_enabled = true

  tags = local.common_tags
}

# Azure Machine Learning Workspace
resource "azurerm_machine_learning_workspace" "main" {
  name                          = "mlw-${var.project_name}-${var.environment}"
  location                      = azurerm_resource_group.main.location
  resource_group_name           = azurerm_resource_group.main.name
  application_insights_id       = azurerm_application_insights.main.id
  key_vault_id                  = azurerm_key_vault.main.id
  storage_account_id            = azurerm_storage_account.ai_foundry.id
  
  public_network_access_enabled = true

  identity {
    type = "SystemAssigned"
  }

  tags = local.common_tags
}

# Container Registry
resource "azurerm_container_registry" "main" {
  name                = "cr${var.project_name}${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.demo_mode ? "Basic" : "Standard"
  admin_enabled       = true

  public_network_access_enabled = true

  tags = local.common_tags
}

# App Service Plan (for demo API)
resource "azurerm_service_plan" "main" {
  count               = var.demo_mode ? 1 : 0
  name                = "asp-${var.project_name}-demo"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "F1"  # Free tier

  tags = local.common_tags
}

# App Service (for demo API)
resource "azurerm_linux_web_app" "main" {
  count               = var.demo_mode ? 1 : 0
  name                = "app-${var.project_name}-demo-${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  service_plan_id     = azurerm_service_plan.main[0].id

  site_config {
    always_on = false
    
    application_stack {
      python_version = "3.11"
    }
  }

  app_settings = {
    "AZURE_OPENAI_ENDPOINT"             = azurerm_cognitive_account.openai.endpoint
    "AZURE_SEARCH_ENDPOINT"             = "https://${azurerm_search_service.main.name}.search.windows.net"
    "AZURE_COMPUTER_VISION_ENDPOINT"    = azurerm_cognitive_account.computer_vision.endpoint
    "AZURE_CONTENT_SAFETY_ENDPOINT"     = azurerm_cognitive_account.content_safety.endpoint
    "AZURE_STORAGE_ACCOUNT"             = azurerm_storage_account.ai_foundry.name
    "AZURE_KEY_VAULT_URL"               = azurerm_key_vault.main.vault_uri
    "APPLICATIONINSIGHTS_CONNECTION_STRING" = azurerm_application_insights.main.connection_string
  }

  tags = local.common_tags
}

# Budget for cost management
resource "azurerm_consumption_budget_resource_group" "main" {
  name              = "budget-${var.project_name}-${var.environment}"
  resource_group_id = azurerm_resource_group.main.id

  amount     = var.demo_mode ? 500 : 2000  # INR
  time_grain = "Monthly"

  time_period {
    start_date = "2025-09-01T00:00:00Z"
    end_date   = "2026-09-01T00:00:00Z"
  }

  notification {
    enabled        = true
    threshold      = 80
    operator       = "GreaterThan"
    threshold_type = "Actual"

    contact_emails = [
      var.alert_email,
    ]
  }

  notification {
    enabled        = true
    threshold      = 95
    operator       = "GreaterThan"
    threshold_type = "Forecasted"

    contact_emails = [
      var.alert_email,
    ]
  }
}

# Store secrets in Key Vault
resource "azurerm_key_vault_secret" "openai_endpoint" {
  name         = "openai-endpoint"
  value        = azurerm_cognitive_account.openai.endpoint
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault.main]
}

resource "azurerm_key_vault_secret" "openai_key" {
  name         = "openai-api-key"
  value        = azurerm_cognitive_account.openai.primary_access_key
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault.main]
}

resource "azurerm_key_vault_secret" "search_endpoint" {
  name         = "search-endpoint"
  value        = "https://${azurerm_search_service.main.name}.search.windows.net"
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault.main]
}

resource "azurerm_key_vault_secret" "search_key" {
  name         = "search-admin-key"
  value        = azurerm_search_service.main.primary_key
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault.main]
}

resource "azurerm_key_vault_secret" "cv_endpoint" {
  name         = "cv-endpoint"
  value        = azurerm_cognitive_account.computer_vision.endpoint
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault.main]
}

resource "azurerm_key_vault_secret" "cv_key" {
  name         = "cv-api-key"
  value        = azurerm_cognitive_account.computer_vision.primary_access_key
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault.main]
}

resource "azurerm_key_vault_secret" "safety_endpoint" {
  name         = "safety-endpoint"
  value        = azurerm_cognitive_account.content_safety.endpoint
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault.main]
}

resource "azurerm_key_vault_secret" "safety_key" {
  name         = "safety-api-key"
  value        = azurerm_cognitive_account.content_safety.primary_access_key
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault.main]
}

# Local values for common tags
locals {
  common_tags = {
    Environment = var.environment
    Project     = var.project_name
    Purpose     = "AI/ML Platform"
    Team        = "MLOps"
    DemoMode    = var.demo_mode ? "true" : "false"
    CreatedBy   = "Terraform"
    CreatedDate = formatdate("YYYY-MM-DD", timestamp())
  }
}