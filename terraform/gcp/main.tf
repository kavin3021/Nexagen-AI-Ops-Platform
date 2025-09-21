# NexaGen AI Ops Platform - GCP Infrastructure
# Terraform configuration for Vertex AI, Gemini, and supporting services

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Random string for unique resource naming
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "container.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "generativelanguage.googleapis.com"
  ])

  service = each.value
  
  disable_dependent_services = false
  disable_on_destroy         = false
}

# Service Account for Vertex AI
resource "google_service_account" "vertex_ai" {
  account_id   = "nexagen-vertex-ai"
  display_name = "NexaGen Vertex AI Service Account"
  description  = "Service account for Vertex AI operations in NexaGen platform"

  depends_on = [google_project_service.apis]
}

# IAM roles for Vertex AI service account
resource "google_project_iam_member" "vertex_ai_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/aiplatform.admin",
    "roles/storage.admin",
    "roles/secretmanager.secretAccessor",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter",
    "roles/ml.admin"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.vertex_ai.email}"

  depends_on = [google_service_account.vertex_ai]
}

# Cloud Storage bucket for Vertex AI staging
resource "google_storage_bucket" "vertex_staging" {
  name          = "${var.project_id}-vertex-staging-${random_string.suffix.result}"
  location      = var.region
  force_destroy = var.demo_mode

  uniform_bucket_level_access = true

  versioning {
    enabled = !var.demo_mode
  }

  lifecycle_rule {
    condition {
      age = var.demo_mode ? 1 : 30
    }
    action {
      type = "Delete"
    }
  }

  labels = local.common_labels

  depends_on = [google_project_service.apis]
}

# Cloud Storage bucket for fine-tuning data
resource "google_storage_bucket" "finetune_data" {
  name          = "${var.project_id}-finetune-data-${random_string.suffix.result}"
  location      = var.region
  force_destroy = var.demo_mode

  uniform_bucket_level_access = true

  versioning {
    enabled = !var.demo_mode
  }

  labels = local.common_labels

  depends_on = [google_project_service.apis]
}

# Cloud Storage bucket for model artifacts
resource "google_storage_bucket" "model_artifacts" {
  name          = "${var.project_id}-model-artifacts-${random_string.suffix.result}"
  location      = var.region
  force_destroy = var.demo_mode

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  labels = local.common_labels

  depends_on = [google_project_service.apis]
}

# Upload training data to storage bucket
resource "google_storage_bucket_object" "training_data" {
  name   = "training-data/train.jsonl"
  bucket = google_storage_bucket.finetune_data.name
  source = "${path.module}/../../data/fine_tune/train.jsonl"

  depends_on = [google_storage_bucket.finetune_data]
}

resource "google_storage_bucket_object" "validation_data" {
  name   = "training-data/validation.jsonl"
  bucket = google_storage_bucket.finetune_data.name
  source = "${path.module}/../../data/fine_tune/validation.jsonl"

  depends_on = [google_storage_bucket.finetune_data]
}

# VPC Network
resource "google_compute_network" "main" {
  name                    = "nexagen-network"
  auto_create_subnetworks = false
  mtu                     = 1460

  depends_on = [google_project_service.apis]
}

# Subnet for Vertex AI workloads
resource "google_compute_subnetwork" "vertex_subnet" {
  name          = "nexagen-vertex-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.main.id

  secondary_ip_range {
    range_name    = "pod-range"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "service-range"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# Firewall rule for internal communication
resource "google_compute_firewall" "internal" {
  name    = "nexagen-internal"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8000", "8080"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/8"]
  target_tags   = ["nexagen"]
}

# Secret Manager secrets
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "nexagen-gemini-api-key"

  replication {
    auto {}
  }

  labels = local.common_labels

  depends_on = [google_project_service.apis]
}

# Cloud Monitoring notification channel
resource "google_monitoring_notification_channel" "email" {
  display_name = "NexaGen Email Alerts"
  type         = "email"

  labels = {
    email_address = var.alert_email
  }

  enabled = true
}

# Budget for cost management
resource "google_billing_budget" "main" {
  count = var.demo_mode ? 1 : 0

  billing_account = var.billing_account_id
  display_name    = "NexaGen Demo Budget"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = var.demo_mode ? "10" : "100"  # $10 for demo, $100 for full
    }
  }

  threshold_rules {
    threshold_percent = 0.8
  }

  threshold_rules {
    threshold_percent = 0.9
  }

  threshold_rules {
    threshold_percent = 1.0
  }

  all_updates_rule {
    monitoring_notification_channels = [
      google_monitoring_notification_channel.email.name
    ]
  }
}

# Cloud Build trigger for CI/CD (optional)
resource "google_cloudbuild_trigger" "main" {
  count = var.enable_cicd ? 1 : 0

  name        = "nexagen-deploy-trigger"
  description = "Deploy NexaGen AI Ops Platform"

  github {
    owner = var.github_owner
    name  = var.github_repo
    push {
      branch = "^main$"
    }
  }

  build {
    step {
      name = "gcr.io/cloud-builders/gcloud"
      args = [
        "ai",
        "models",
        "list",
        "--region=${var.region}"
      ]
    }

    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "build",
        "-t",
        "gcr.io/${var.project_id}/nexagen-api:$COMMIT_SHA",
        "."
      ]
    }

    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "push",
        "gcr.io/${var.project_id}/nexagen-api:$COMMIT_SHA"
      ]
    }
  }

  depends_on = [google_project_service.apis]
}

# GKE Cluster for advanced deployments (optional)
resource "google_container_cluster" "main" {
  count = var.enable_gke ? 1 : 0

  name     = "nexagen-cluster"
  location = var.zone

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.main.name
  subnetwork = google_compute_subnetwork.vertex_subnet.name

  ip_allocation_policy {
    cluster_secondary_range_name  = "pod-range"
    services_secondary_range_name = "service-range"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  network_policy {
    enabled = true
  }

  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
    network_policy_config {
      disabled = false
    }
  }

  depends_on = [google_project_service.apis]
}

# Node pool for GKE cluster
resource "google_container_node_pool" "main" {
  count = var.enable_gke ? 1 : 0

  name       = "nexagen-node-pool"
  location   = var.zone
  cluster    = google_container_cluster.main[0].name
  node_count = var.demo_mode ? 1 : 2

  node_config {
    preemptible  = var.demo_mode
    machine_type = var.demo_mode ? "e2-small" : "e2-medium"

    service_account = google_service_account.vertex_ai.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = local.common_labels
    tags   = ["nexagen"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  autoscaling {
    min_node_count = var.demo_mode ? 0 : 1
    max_node_count = var.demo_mode ? 2 : 5
  }
}

# Cloud Run service for API deployment (alternative to GKE)
resource "google_cloud_run_v2_service" "api" {
  count = var.enable_cloud_run ? 1 : 0

  name     = "nexagen-api"
  location = var.region

  template {
    containers {
      image = "gcr.io/${var.project_id}/nexagen-api:latest"

      ports {
        container_port = 8000
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "VERTEX_AI_LOCATION"
        value = var.region
      }

      resources {
        limits = {
          cpu    = var.demo_mode ? "1" : "2"
          memory = var.demo_mode ? "512Mi" : "2Gi"
        }
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = var.demo_mode ? 2 : 10
    }

    service_account = google_service_account.vertex_ai.email
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [google_project_service.apis]
}

# IAM policy for Cloud Run (allow public access for demo)
resource "google_cloud_run_service_iam_member" "public" {
  count = var.enable_cloud_run && var.demo_mode ? 1 : 0

  location = google_cloud_run_v2_service.api[0].location
  project  = google_cloud_run_v2_service.api[0].project
  service  = google_cloud_run_v2_service.api[0].name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Vertex AI Dataset for fine-tuning
resource "google_vertex_ai_dataset" "text_dataset" {
  count = var.enable_vertex_ai_dataset ? 1 : 0

  display_name   = "nexagen-text-dataset"
  metadata_schema_uri = "gs://google-cloud-aiplatform/schema/dataset/metadata/text_1.0.0.yaml"
  region         = var.region

  labels = local.common_labels

  depends_on = [google_project_service.apis]
}

# Log-based metric for monitoring
resource "google_logging_metric" "error_rate" {
  name   = "nexagen_error_rate"
  filter = <<-EOT
    resource.type="cloud_run_revision"
    resource.labels.service_name="nexagen-api"
    severity>=ERROR
  EOT

  metric_descriptor {
    metric_kind = "GAUGE"
    value_type  = "INT64"
    display_name = "NexaGen Error Rate"
  }

  value_extractor = "EXTRACT(jsonPayload.error_count)"

  label_extractors = {
    "service_name" = "EXTRACT(resource.labels.service_name)"
  }

  depends_on = [google_project_service.apis]
}

# Monitoring alert policy
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "NexaGen High Error Rate"
  combiner     = "OR"

  conditions {
    display_name = "Error rate too high"

    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/nexagen_error_rate\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [
    google_monitoring_notification_channel.email.name
  ]

  depends_on = [google_logging_metric.error_rate]
}

# Monitoring dashboard
resource "google_monitoring_dashboard" "main" {
  dashboard_json = jsonencode({
    displayName = "NexaGen AI Ops Platform"
    mosaicLayout = {
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "API Request Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"run.googleapis.com/request_count\""
                  }
                }
              }]
            }
          }
        },
        {
          width  = 6
          height = 4
          yPos   = 0
          xPos   = 6
          widget = {
            title = "Error Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/nexagen_error_rate\""
                  }
                }
              }]
            }
          }
        },
        {
          width  = 6
          height = 4
          yPos   = 4
          widget = {
            title = "Response Latency"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"run.googleapis.com/request_latencies\""
                  }
                }
              }]
            }
          }
        },
        {
          width  = 6
          height = 4
          yPos   = 4
          xPos   = 6
          widget = {
            title = "Memory Usage"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"run.googleapis.com/container/memory/utilizations\""
                  }
                }
              }]
            }
          }
        }
      ]
    }
  })

  depends_on = [google_project_service.apis]
}

# Local values for common labels
locals {
  common_labels = {
    environment = var.environment
    project     = var.project_name
    purpose     = "ai-ml-platform"
    team        = "mlops"
    demo_mode   = var.demo_mode ? "true" : "false"
    created_by  = "terraform"
  }
}