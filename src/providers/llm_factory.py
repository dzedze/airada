# llm_factory.py

import os

from pydantic import SecretStr
from openai import OpenAI
from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
)
from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv

# Load the .secrets file
env_path = Path(__file__).parent.parent.parent / ".secrets"
load_dotenv(dotenv_path=env_path)

# Access environment variables (lazy loading - only when needed)
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_OPENAI_MODEL = "gpt-4o"


def _get_api_key() -> str:
    """Get OpenAI API key from environment, with error if not set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is not set")
    return api_key


def create_openai_client(
    api_key: str | None = None,
) -> OpenAI:
    """
    Create and return a configured OpenAI client.
    """
    key = api_key or _get_api_key()
    return OpenAI(
        api_key=key,
    )


def create_embedding_function(
    api_key: str | None = None,
) -> OpenAIEmbeddingFunction:
    """
    Create and return a Chroma-compatible OpenAI embedding function.

    Uses environment variable for API key storage (as recommended by ChromaDB)
    to avoid deprecation warnings.
    """
    key = api_key or _get_api_key()

    # Set the API key in environment for ChromaDB to use
    # (using api_key_env_var parameter instead of direct api_key)
    os.environ["OPENAI_API_KEY"] = key

    ef = OpenAIEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        api_key_env_var="OPENAI_API_KEY",
    )
    ef.client = create_openai_client(api_key=key)
    return ef


def create_chat_agent(
    api_key: str | None = None,
    model: str | None = None,
) -> ChatOpenAI:
    """
    Create and return a configured ChatOpenAI.
    """
    return ChatOpenAI(
        model=model or CHAT_OPENAI_MODEL,
        api_key=SecretStr(api_key) if api_key else None,
    )
