import os

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq


def get_chat_model():
    """Create an LLM using the configured inference provider."""

    provider = os.getenv("MODEL_PROVIDER", "ollama").lower()
    max_output = int(os.getenv("MAX_OUTPUT_TOKENS", "384"))

    if provider == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://127.0.0.1:11434",
            ),
            temperature=0,
            num_predict=max_output,
        )

    if provider == "groq":
        return ChatGroq(
            model=os.environ["GROQ_MODEL"],
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0,
            max_tokens=max_output,
            max_retries=0,
        )

    raise ValueError(f"Unsupported model provider: {provider}")
