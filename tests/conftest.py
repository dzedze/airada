"""
Shared pytest fixtures and configuration.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_papers_csv(temp_dir):
    """Create a sample papers.csv for testing."""
    data = {
        "title": [
            "Multi-agent Reinforcement Learning Framework",
            "Retrieval Augmented Generation for LLMs",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "A Simple Paper About Cats",
        ],
        "abstract": [
            "This paper describes an autonomous agent system using tool use and planning mechanisms.",
            "We present a rag-based approach to enhance language model performance with memory-augmented retrieval.",
            "We introduce BERT, a large language model trained using masked language modeling.",
            "This paper discusses feline behavior and care.",
        ],
        "url_abs": [
            "http://arxiv.org/abs/2301.001",
            "http://arxiv.org/abs/2301.002",
            "http://arxiv.org/abs/1810.04805",
            "http://arxiv.org/abs/2301.999",
        ],
        "url_pdf": [
            "http://arxiv.org/pdf/2301.001.pdf",
            "http://arxiv.org/pdf/2301.002.pdf",
            "http://arxiv.org/pdf/1810.04805.pdf",
            "http://arxiv.org/pdf/2301.999.pdf",
        ],
        "arxiv_id": [
            "2301.001",
            "2301.002",
            "1810.04805",
            "2301.999",
        ],
    }
    csv_path = temp_dir / "papers.csv"
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_papers_dataframe():
    """Create a sample papers DataFrame for testing."""
    return pd.DataFrame(
        {
            "title": [
                "Multi-agent Reinforcement Learning Framework",
                "Retrieval Augmented Generation for LLMs",
                "BERT: Pre-training of Deep Bidirectional Transformers",
            ],
            "abstract": [
                "This paper describes an autonomous agent system using tool use and planning mechanisms.",
                "We present a rag-based approach to enhance language model performance with memory-augmented retrieval.",
                "We introduce BERT, a large language model trained using masked language modeling.",
            ],
            "url_abs": [
                "http://arxiv.org/abs/2301.001",
                "http://arxiv.org/abs/2301.002",
                "http://arxiv.org/pdf/1810.04805",
            ],
            "url_pdf": [
                "http://arxiv.org/pdf/2301.001.pdf",
                "http://arxiv.org/pdf/2301.002.pdf",
                "http://arxiv.org/pdf/1810.04805.pdf",
            ],
            "arxiv_id": [
                "2301.001",
                "2301.002",
                "1810.04805",
            ],
        }
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock = MagicMock()
    mock.models.list.return_value = MagicMock(data=[])
    return mock


@pytest.fixture
def mock_embedding_function():
    """Create a mock embedding function for Chroma."""
    mock = MagicMock()
    mock.client = MagicMock()
    return mock


@pytest.fixture
def mock_chroma_collection():
    """Create a mock Chroma collection."""
    mock = MagicMock()
    mock.count.return_value = 0
    mock.upsert = MagicMock()
    mock.query = MagicMock(
        return_value={
            "ids": [["2301.001", "2301.002"]],
            "metadatas": [
                [
                    {"title": "Test Paper 1"},
                    {"title": "Test Paper 2"},
                ]
            ],
            "documents": [["Test doc 1", "Test doc 2"]],
            "distances": [[0.1, 0.2]],
        }
    )
    return mock
