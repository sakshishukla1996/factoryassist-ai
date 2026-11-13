import os

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from langchain_core.documents import Document


def load_azure_documents() -> list[Document]:
    """Load manufacturing documents from Azure Blob Storage."""

    account_name = os.environ["AZURE_STORAGE_ACCOUNT"]

    container_name = os.getenv(
        "AZURE_BLOB_CONTAINER",
        "manufacturing-docs",
    )

    account_url = (
        f"https://{account_name}.blob.core.windows.net"
    )

    credential = DefaultAzureCredential()

    blob_service = BlobServiceClient(
        account_url=account_url,
        credential=credential,
    )

    container_client = blob_service.get_container_client(
        container_name
    )

    documents = []

    for blob in container_client.list_blobs():
        if not blob.name.endswith(".txt"):
            continue

        blob_client = container_client.get_blob_client(
            blob.name
        )

        content = (
            blob_client
            .download_blob()
            .readall()
            .decode("utf-8")
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": f"data/{blob.name}",
                    "storage_backend": "azure",
                    "blob_name": blob.name,
                },
            )
        )

    if not documents:
        raise ValueError(
            "No manufacturing documents found in Azure."
        )

    print(f"Loaded {len(documents)} documents from Azure")

    return documents
