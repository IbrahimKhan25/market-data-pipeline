output "storage_account_name" {
  value = azurerm_storage_account.lake.name
}

output "container_name" {
  value = azurerm_storage_container.market_data.name
}

output "connection_string" {
  description = "Set as the AZURE_STORAGE_CONNECTION_STRING secret for the pipeline."
  value       = azurerm_storage_account.lake.primary_connection_string
  sensitive   = true
}
