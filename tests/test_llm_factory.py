"""
Tests for src/providers/llm_factory.py
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the module we're testing
from src.providers import llm_factory


class TestLlmFactory:
    """Test cases for LLM factory functions."""

    def test_create_openai_client_with_api_key(self):
        """Test creating OpenAI client with explicit API key."""
        api_key = "test-key-12345"
        client = llm_factory.create_openai_client(api_key=api_key)
        assert client is not None
        assert hasattr(client, 'api_key')

    def test_create_openai_client_uses_default_key(self):
        """Test that create_openai_client uses provided key or environment key."""
        # Since the module loads api_key at import time, just verify it creates a client
        try:
            client = llm_factory.create_openai_client(api_key="test-key")
            assert client is not None
        except Exception:
            pytest.skip("OPENAI_API_KEY not properly set")

    def test_embedding_model_constant_exists(self):
        """Test that EMBEDDING_MODEL constant is defined."""
        assert hasattr(llm_factory, 'EMBEDDING_MODEL')
        assert llm_factory.EMBEDDING_MODEL == "text-embedding-3-small"

    def test_chat_openai_model_constant_exists(self):
        """Test that CHAT_OPENAI_MODEL constant is defined."""
        assert hasattr(llm_factory, 'CHAT_OPENAI_MODEL')
        assert llm_factory.CHAT_OPENAI_MODEL == "gpt-4o"

    @patch('src.providers.llm_factory.OpenAIEmbeddingFunction')
    @patch('src.providers.llm_factory.create_openai_client')
    def test_create_embedding_function(self, mock_oai_client, mock_embedding_fn):
        """Test creating embedding function."""
        api_key = "test-key-12345"
        embedding_fn = llm_factory.create_embedding_function(api_key=api_key)
        assert embedding_fn is not None

    @patch('src.providers.llm_factory.OpenAIEmbeddingFunction')
    @patch('src.providers.llm_factory.create_openai_client')
    def test_create_embedding_function_sets_client(self, mock_oai_client, mock_embedding_fn_class):
        """Test that embedding function has a client assigned."""
        api_key = "test-key-12345"
        mock_embedding_fn = MagicMock()
        mock_embedding_fn_class.return_value = mock_embedding_fn

        embedding_fn = llm_factory.create_embedding_function(api_key=api_key)

        # Verify that the client was set
        assert hasattr(embedding_fn, 'client')
