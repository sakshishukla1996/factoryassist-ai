import os
import tempfile
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


INDEX_FILES = ("index.faiss", "index.pkl")


def download_azure_index() -> Path:
    """Download the trusted FAISS index from private Azure Blob Storage."""

    account = os.environ["AZURE_STORAGE_ACCOUNT"]
    container = os.getenv("AZURE_FAISS_CONTAINER", "faiss-index")

    cache_dir = Path(
        os.getenv(
            "AZURE_FAISS_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "factoryassist" / "faiss"),
        )
    )

    cache_dir.mkdir(parents=True, exist_ok=True)

    credential = DefaultAzureCredential()

    blob_service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=credential,
    )

    for filename in INDEX_FILES:
        destination = cache_dir / filename

        # Avoid downloading the same file repeatedly.
        if destination.exists() and destination.stat().st_size > 0:
            continue

        print(f"Downloading FAISS file: {filename}")

        blob_client = blob_service.get_blob_client(
            container=container,
            blob=filename,
        )

        temporary_file = destination.with_name(
            destination.name + ".download"
        )

        try:
            with temporary_file.open("wb") as file:
                blob_client.download_blob().readinto(file)

            if temporary_file.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty file: {filename}")

            temporary_file.replace(destination)

        except Exception:
            temporary_file.unlink(missing_ok=True)
            raise

    return cache_dir
