import os
from dotenv import load_dotenv

from factoryassist.rag.azure_loader import (
    load_azure_documents,)

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data")
INDEX_DIR = Path("artifacts/faiss")


def load_documents():
    load_dotenv()

    if os.getenv("DOCUMENT_SOURCE", "local") == "azure":
        return load_azure_documents()
    documents = []

    for path in DATA_DIR.rglob("*.txt"):
        loader = TextLoader(str(path), encoding="utf-8")
        documents.extend(loader.load())

    if not documents:
        raise ValueError("No documents found in data/")

    return documents


def build_index():
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    print(f"FAISS index saved to {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
