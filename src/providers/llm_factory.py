# llm_factory.py

import os
from openai import OpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv

# Load the .secrets file
env_path = Path(__file__).parent.parent.parent / ".secrets"
load_dotenv(dotenv_path=env_path)

# Access your environment variables
api_key = os.getenv("OPENAI_API_KEY")
print("api_key:", api_key[:3])

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_OPENAI_MODEL = "gpt-4o"


def create_openai_client(
    api_key: str | None = None,
) -> OpenAI:
    """
    Create and return a configured OpenAI client.
    """
    return OpenAI(
        api_key=api_key or api_key,
    )


def create_embedding_function(
    api_key: str | None = None,
) -> OpenAIEmbeddingFunction:
    """
    Create and return a Chroma-compatible OpenAI embedding function.
    """
    ef = OpenAIEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        api_key=api_key or api_key,
    )
    ef._client = create_openai_client(api_key=api_key)
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
        api_key=api_key or api_key,
    )
