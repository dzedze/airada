"""
Tests for src/data/03_ingest_data.py
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

# Import the ingest_data module directly
from importlib import import_module
ingest_data = import_module('src.data.03_ingest_data')


class TestIngestData:
    """Test cases for ingest_data module."""

    def test_constants_defined(self):
        """Test that required constants are defined."""
        assert hasattr(ingest_data, 'COLLECTION_NAME')
        assert hasattr(ingest_data, 'BATCH_SIZE')
        assert hasattr(ingest_data, 'CSV_PATH')
        assert hasattr(ingest_data, 'CHROMA_DB_PATH')

    def test_batch_size_valid(self):
        """Test that BATCH_SIZE is a positive integer."""
        assert isinstance(ingest_data.BATCH_SIZE, int)
        assert ingest_data.BATCH_SIZE > 0

    def test_collection_name_valid(self):
        """Test that COLLECTION_NAME is defined."""
        assert isinstance(ingest_data.COLLECTION_NAME, str)
        assert len(ingest_data.COLLECTION_NAME) > 0

    def test_build_document_combines_title_and_abstract(self):
        """Test that build_document correctly combines title and abstract."""
        row = pd.Series({
            "title": "Test Paper Title",
            "abstract": "Test abstract content"
        })

        doc = ingest_data.build_document(row)

        assert "Test Paper Title" in doc
        assert "Test abstract content" in doc
        assert "Title:" in doc
        assert "Abstract:" in doc

    def test_build_document_handles_missing_title(self):
        """Test that build_document handles missing title gracefully."""
        row = pd.Series({
            "title": None,
            "abstract": "Test abstract"
        })

        doc = ingest_data.build_document(row)
        assert isinstance(doc, str)
        assert "Test abstract" in doc

    def test_build_document_handles_missing_abstract(self):
        """Test that build_document handles missing abstract gracefully."""
        row = pd.Series({
            "title": "Test Title",
            "abstract": None
        })

        doc = ingest_data.build_document(row)
        assert isinstance(doc, str)
        assert "Test Title" in doc

    def test_build_document_strips_whitespace(self):
        """Test that build_document strips whitespace."""
        row = pd.Series({
            "title": "  Title with spaces  ",
            "abstract": "  Abstract with spaces  "
        })

        doc = ingest_data.build_document(row)
        assert "  Title" not in doc
        assert "  Abstract" not in doc

    def test_build_metadata_includes_required_fields(self):
        """Test that build_metadata includes required metadata fields."""
        row = pd.Series({
            "title": "Test Paper",
            "arxiv_id": "2301.001",
            "url_abs": "http://arxiv.org/abs/2301.001",
            "url_pdf": "http://arxiv.org/pdf/2301.001.pdf"
        })

        metadata = ingest_data.build_metadata(row)

        required_fields = ["title", "arxiv_id", "url_abs", "url_pdf"]
        for field in required_fields:
            assert field in metadata

    def test_build_metadata_handles_missing_fields(self):
        """Test that build_metadata handles missing fields gracefully."""
        row = pd.Series({
            "title": "Test Paper",
            # Missing arxiv_id, url_abs, url_pdf
        })

        metadata = ingest_data.build_metadata(row)

        assert isinstance(metadata, dict)
        assert "title" in metadata

    def test_build_metadata_strips_whitespace(self):
        """Test that build_metadata strips whitespace from fields."""
        row = pd.Series({
            "title": "  Title with spaces  ",
            "arxiv_id": "  2301.001  ",
            "url_abs": "  http://arxiv.org/abs/2301.001  ",
            "url_pdf": "  http://arxiv.org/pdf/2301.001.pdf  "
        })

        metadata = ingest_data.build_metadata(row)

        assert metadata["title"] == "Title with spaces"
        assert metadata["arxiv_id"] == "2301.001"

    @patch('chromadb.PersistentClient')
    @patch('src.providers.llm_factory.create_embedding_function')
    @patch('pandas.read_csv')
    def test_ingest_loads_csv(self, mock_read_csv, mock_embedding, mock_client):
        """Test that ingest loads CSV from defined path."""
        mock_df = pd.DataFrame({
            "title": ["Test Paper"],
            "abstract": ["Test abstract"],
            "url_abs": ["url"],
            "url_pdf": ["pdf"],
            "arxiv_id": ["id"]
        })
        mock_read_csv.return_value = mock_df

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        # Mock embedding function
        mock_embedding.return_value = MagicMock()

        ingest_data.ingest()

        mock_read_csv.assert_called_once()

    @patch('chromadb.PersistentClient')
    @patch('src.providers.llm_factory.create_embedding_function')
    @patch('pandas.read_csv')
    def test_ingest_validates_required_columns(self, mock_read_csv, mock_embedding, mock_client):
        """Test that ingest validates required columns exist."""
        # DataFrame missing required columns
        mock_df = pd.DataFrame({
            "wrong_col1": ["value1"],
            "wrong_col2": ["value2"]
        })
        mock_read_csv.return_value = mock_df

        # Mock chromadb
        mock_collection = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        mock_embedding.return_value = MagicMock()

        with pytest.raises(ValueError, match="missing required columns"):
            ingest_data.ingest()

    @patch('chromadb.PersistentClient')
    @patch('src.providers.llm_factory.create_embedding_function')
    @patch('pandas.read_csv')
    def test_ingest_drops_empty_rows(self, mock_read_csv, mock_embedding, mock_client):
        """Test that ingest removes rows with empty title/abstract."""
        mock_df = pd.DataFrame({
            "title": ["Valid Paper", None, "Another Paper"],
            "abstract": ["Abstract 1", "Abstract 2", None],
            "url_abs": ["url1", "url2", "url3"],
            "url_pdf": ["pdf1", "pdf2", "pdf3"],
            "arxiv_id": ["id1", "id2", "id3"]
        })
        mock_read_csv.return_value = mock_df

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        mock_embedding.return_value = MagicMock()

        ingest_data.ingest()

        # Verify upsert was called (indicating documents were created)
        mock_collection.upsert.assert_called()

    @patch('chromadb.PersistentClient')
    @patch('src.providers.llm_factory.create_embedding_function')
    @patch('pandas.read_csv')
    def test_ingest_creates_embeddings_in_batches(self, mock_read_csv, mock_embedding, mock_client):
        """Test that ingest processes documents in batches."""
        # Create DataFrame with more rows than one batch
        num_papers = 250  # > BATCH_SIZE (100)
        mock_df = pd.DataFrame({
            "title": [f"Paper {i}" for i in range(num_papers)],
            "abstract": [f"Abstract {i}" for i in range(num_papers)],
            "url_abs": [f"url{i}" for i in range(num_papers)],
            "url_pdf": [f"pdf{i}" for i in range(num_papers)],
            "arxiv_id": [f"id{i}" for i in range(num_papers)]
        })
        mock_read_csv.return_value = mock_df

        # Mock chromadb
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        mock_embedding.return_value = MagicMock()

        ingest_data.ingest()

        # Verify upsert was called multiple times (batches)
        assert mock_collection.upsert.call_count >= 2
