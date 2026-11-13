
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Read Langfuse credentials from the root .env file.
load_dotenv()


def get_langfuse_handler() -> CallbackHandler:
    """Create the callback handler for LangChain tracing."""
    return CallbackHandler()


def verify_langfuse_connection() -> bool:
    """Check that our API credentials are valid."""
    return get_client().auth_check()


def flush_langfuse() -> None:
    """Send any queued traces before a short script exits."""
    get_client().flush()
