# NexaGen AI Ops Platform Variables - Azure
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "nexagen"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "East US"
}

variable "demo_mode" {
  description = "Enable demo mode with minimal resources and costs"
  type        = bool
  default     = true
}

variable "alert_email" {
  description = "Email address for alerts and notifications"
  type        = string
  default     = "admin@example.com"
}

# Azure OpenAI Configuration
variable "gpt_deployment_name" {
  description = "Name for GPT deployment"
  type        = string
  default     = "nexagen-gpt-flow"
}

variable "embedding_deployment_name" {
  description = "Name for embedding deployment"
  type        = string
  default     = "nexagen-embed"
}

variable "openai_model_capacity" {
  description = "Capacity for OpenAI models"
  type        = number
  default     = 1
}

# Resource Configuration
variable "app_service_sku" {
  description = "App Service plan SKU"
  type        = string
  default     = "F1"  # Free tier for demo
}

variable "storage_account_tier" {
  description = "Storage account tier"
  type        = string
  default     = "Standard"
}

variable "storage_replication_type" {
  description = "Storage replication type"
  type        = string
  default     = "LRS"
}

variable "search_service_sku" {
  description = "Azure Search service SKU"
  type        = string
  default     = "free"  # Free tier for demo
}

variable "key_vault_sku" {
  description = "Key Vault SKU"
  type        = string
  default     = "standard"
}

# Budget and Cost Management
variable "monthly_budget_limit" {
  description = "Monthly budget limit in local currency"
  type        = number
  default     = 500  # ₹500 for demo mode
}

variable "budget_alert_threshold_80" {
  description = "Budget alert threshold at 80%"
  type        = number
  default     = 0.8
}

variable "budget_alert_threshold_95" {
  description = "Budget alert threshold at 95%"
  type        = number
  default     = 0.95
}

# Network Configuration
variable "vnet_address_space" {
  description = "Virtual network address space"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "subnet_ai_services_prefix" {
  description = "AI services subnet address prefix"
  type        = list(string)
  default     = ["10.0.1.0/24"]
}

variable "subnet_compute_prefix" {
  description = "Compute subnet address prefix"
  type        = list(string)
  default     = ["10.0.2.0/24"]
}

# Security Configuration
variable "enable_private_endpoints" {
  description = "Enable private endpoints for services"
  type        = bool
  default     = false  # Disabled for demo to reduce cost
}

variable "enable_network_security_group" {
  description = "Enable network security groups"
  type        = bool
  default     = true
}

variable "allowed_ip_ranges" {
  description = "Allowed IP ranges for access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Open for demo, restrict in production
}

# Monitoring and Logging
variable "log_analytics_retention_days" {
  description = "Log Analytics workspace retention in days"
  type        = number
  default     = 30  # Minimum for demo
}

variable "app_insights_retention_days" {
  description = "Application Insights retention in days"
  type        = number
  default     = 30
}

variable "enable_diagnostic_settings" {
  description = "Enable diagnostic settings for resources"
  type        = bool
  default     = true
}

# Cognitive Services Configuration
variable "cognitive_services_sku" {
  description = "SKU for Cognitive Services (Computer Vision, Content Safety)"
  type        = string
  default     = "F0"  # Free tier
}

variable "openai_sku" {
  description = "SKU for Azure OpenAI"
  type        = string
  default     = "S0"  # Standard tier (minimum for OpenAI)
}

# Machine Learning Configuration
variable "ml_workspace_sku" {
  description = "Machine Learning workspace SKU"
  type        = string
  default     = "Basic"
}

variable "enable_ml_compute_instance" {
  description = "Enable ML compute instance"
  type        = bool
  default     = false  # Disabled for demo to save costs
}

variable "ml_compute_instance_size" {
  description = "ML compute instance size"
  type        = string
  default     = "Standard_DS3_v2"
}

# Container Registry
variable "acr_sku" {
  description = "Container Registry SKU"
  type        = string
  default     = "Basic"
}

variable "acr_admin_enabled" {
  description = "Enable admin user for Container Registry"
  type        = bool
  default     = true
}

# Tags
variable "default_tags" {
  description = "Default tags to apply to resources"
  type        = map(string)
  default     = {
    Environment = "dev"
    Project     = "nexagen"
    Purpose     = "ai-ml-platform"
    Team        = "mlops"
    CreatedBy   = "terraform"
  }
}

# Feature Flags
variable "enable_app_service" {
  description = "Enable App Service for demo API"
  type        = bool
  default     = true
}

variable "enable_container_registry" {
  description = "Enable Azure Container Registry"
  type        = bool
  default     = true
}

variable "enable_ml_workspace" {
  description = "Enable Azure ML workspace"
  type        = bool
  default     = true
}

variable "enable_search_service" {
  description = "Enable Azure Search service"
  type        = bool
  default     = true
}

variable "enable_key_vault" {
  description = "Enable Azure Key Vault"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable monitoring and alerting"
  type        = bool
  default     = true
}

# Development and Testing
variable "enable_development_features" {
  description = "Enable development-specific features"
  type        = bool
  default     = true
}

variable "skip_provider_registration" {
  description = "Skip provider registration (useful for some subscriptions)"
  type        = bool
  default     = false
}

# Backup and Recovery
variable "enable_backup" {
  description = "Enable backup for critical resources"
  type        = bool
  default     = false  # Disabled for demo
}

variable "backup_retention_days" {
  description = "Backup retention in days"
  type        = number
  default     = 7
}

# Performance and Scaling
variable "enable_autoscaling" {
  description = "Enable autoscaling for applicable resources"
  type        = bool
  default     = false  # Keep simple for demo
}

variable "min_capacity" {
  description = "Minimum capacity for autoscaling"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum capacity for autoscaling"
  type        = number
  default     = 3
}

# Database Configuration (if needed)
variable "enable_database" {
  description = "Enable database resources"
  type        = bool
  default     = false  # Not needed for demo
}

variable "database_sku" {
  description = "Database SKU"
  type        = string
  default     = "Basic"
}

# CDN Configuration
variable "enable_cdn" {
  description = "Enable CDN for static assets"
  type        = bool
  default     = false  # Not needed for demo
}

variable "cdn_sku" {
  description = "CDN SKU"
  type        = string
  default     = "Standard_Microsoft"
}

# Load Balancer Configuration
variable "enable_load_balancer" {
  description = "Enable load balancer"
  type        = bool
  default     = false  # Not needed for demo
}

variable "load_balancer_sku" {
  description = "Load balancer SKU"
  type        = string
  default     = "Basic"
}

# Custom Domain Configuration
variable "custom_domain" {
  description = "Custom domain for services"
  type        = string
  default     = ""
}

variable "enable_ssl_certificate" {
  description = "Enable SSL certificate"
  type        = bool
  default     = false
}

# API Management
variable "enable_api_management" {
  description = "Enable API Management service"
  type        = bool
  default     = false  # Expensive, skip for demo
}

variable "api_management_sku" {
  description = "API Management SKU"
  type        = string
  default     = "Developer"
}

# Data Factory (for advanced data pipelines)
variable "enable_data_factory" {
  description = "Enable Data Factory for ETL pipelines"
  type        = bool
  default     = false  # Not needed for basic demo
}

# Event Hub (for streaming data)
variable "enable_event_hub" {
  description = "Enable Event Hub for streaming"
  type        = bool
  default     = false  # Not needed for basic demo
}

# Service Bus (for messaging)
variable "enable_service_bus" {
  description = "Enable Service Bus for messaging"
  type        = bool
  default     = false  # Not needed for basic demo
}

# Redis Cache
variable "enable_redis_cache" {
  description = "Enable Redis cache"
  type        = bool
  default     = false  # Can be expensive, skip for demo
}

variable "redis_cache_sku" {
  description = "Redis cache SKU"
  type        = string
  default     = "Basic"
}

# Logic Apps (for workflow automation)
variable "enable_logic_apps" {
  description = "Enable Logic Apps for workflows"
  type        = bool
  default     = false  # Not needed for demo
}

# Functions App (for serverless functions)
variable "enable_functions_app" {
  description = "Enable Azure Functions"
  type        = bool
  default     = false  # App Service is sufficient for demo
}

# Notification Hub
variable "enable_notification_hub" {
  description = "Enable Notification Hub for push notifications"
  type        = bool
  default     = false  # Not needed for demo
}

# Time Series Insights
variable "enable_time_series_insights" {
  description = "Enable Time Series Insights for IoT data"
  type        = bool
  default     = false  # Not applicable to demo
}

# Purview (for data governance)
variable "enable_purview" {
  description = "Enable Microsoft Purview for data governance"
  type        = bool
  default     = false  # Expensive and not needed for demo
}