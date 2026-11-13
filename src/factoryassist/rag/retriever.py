import os

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache

from factoryassist.rag.azure_index import download_azure_index

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def load_vectorstore():
    """Load FAISS from local disk or Azure Blob Storage."""

    index_source = os.getenv("INDEX_SOURCE", "local").lower()

    if index_source == "local":
        index_dir = Path(
            os.getenv("FAISS_INDEX_PATH", "artifacts/faiss")
        )

    elif index_source == "azure":
        index_dir = download_azure_index()

    else:
        raise ValueError(
            f"Unsupported INDEX_SOURCE: {index_source}"
        )

    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_dir}"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )

    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore

def get_retriever(k: int = 3):
    """Return a retriever with the requested number of results."""

    vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )



def main():
    question = (
        "What causes error E-2047 in machine M-204, "
        "and how should maintenance personnel respond?"
    )

    retriever = get_retriever(k=3)
    documents = retriever.invoke(question)

    print(f"\nQUESTION: {question}")
    print(f"\nRETRIEVED {len(documents)} CHUNKS\n")

    for index, doc in enumerate(documents, start=1):
        print(f"{'=' * 55}")
        print(f"RESULT {index}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content:\n{doc.page_content}")
        print()


if __name__ == "__main__":
    main()
