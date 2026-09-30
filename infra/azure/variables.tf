variable "project" {
  description = "Short project name used in resource names."
  type        = string
  default     = "mktpipe"
}

variable "environment" {
  description = "Deployment environment (dev, prod)."
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be dev or prod."
  }
}

variable "location" {
  description = "Azure region."
  type        = string
  default     = "uksouth"
}

variable "container_name" {
  description = "Blob container for published data (matches azure.container in config/pipeline.yaml)."
  type        = string
  default     = "market-data"
}

variable "blob_prefix" {
  description = "Blob prefix used by the publisher (matches azure.prefix)."
  type        = string
  default     = "market-pipeline"
}
