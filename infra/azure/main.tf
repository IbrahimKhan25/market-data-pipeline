terraform {
  required_version = ">= 1.6"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "this" {
  name     = "rg-${var.project}-${var.environment}"
  location = var.location
  tags     = local.tags
}

# ADLS Gen2 (hierarchical namespace), so Synapse / Fabric / Databricks can query the gold layer directly.
resource "azurerm_storage_account" "lake" {
  name                            = substr(replace("st${var.project}${var.environment}", "-", ""), 0, 24)
  resource_group_name             = azurerm_resource_group.this.name
  location                        = azurerm_resource_group.this.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  account_kind                    = "StorageV2"
  is_hns_enabled                  = true
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  https_traffic_only_enabled      = true
  tags                            = local.tags

  blob_properties {
    delete_retention_policy {
      days = 7
    }
  }
}

resource "azurerm_storage_container" "market_data" {
  name                  = var.container_name
  storage_account_id    = azurerm_storage_account.lake.id
  container_access_type = "private"
}

# Dated snapshots cool after 30 days and are deleted after a year; `latest/` stays hot.
resource "azurerm_storage_management_policy" "snapshots" {
  storage_account_id = azurerm_storage_account.lake.id

  rule {
    name    = "age-out-gold-snapshots"
    enabled = true
    filters {
      blob_types   = ["blockBlob"]
      prefix_match = ["${var.container_name}/${var.blob_prefix}/gold/"]
    }
    actions {
      base_blob {
        tier_to_cool_after_days_since_modification_greater_than = 30
        delete_after_days_since_modification_greater_than       = 365
      }
    }
  }
}

locals {
  tags = {
    project     = var.project
    environment = var.environment
    managed_by  = "terraform"
  }
}
