from azure.storage.blob import BlobServiceClient
import os

# === CONFIG ===
CONNECTION_STRING = os.environ.get("AZURE_CONNECTION_STRING")
CONTAINER_NAME = "ibrahimmarketdata"  
BLOB_NAME = "stage_clean.csv"           # how it will be named in Azure
LOCAL_FILE_PATH = "stage_clean.csv"     # file we created from transform.py


def upload_to_azure_blob():
    if not os.path.exists(LOCAL_FILE_PATH):
        raise FileNotFoundError(f"{LOCAL_FILE_PATH} not found. Run transform.py first.")

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

    # Get container client
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Ensure container exists (won't error if it already exists)
    try:
        container_client.create_container()
    except Exception:
        pass  # probably already exists

    # Get blob client
    blob_client = container_client.get_blob_client(BLOB_NAME)

    # Upload the file
    with open(LOCAL_FILE_PATH, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"Uploaded {LOCAL_FILE_PATH} to Azure Blob as {BLOB_NAME} in container '{CONTAINER_NAME}'.")


if __name__ == "__main__":
    upload_to_azure_blob()
