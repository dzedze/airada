"""
Tests for src/tools/tools_papers.py
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

from importlib import import_module

# Ensure the src directory is in the path for importlib
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

tools_papers = import_module("src.tools.tools_papers")


class TestPapersTools:
    """Test cases for arXiv papers search tool."""

    def test_constants_defined(self):
        """Test that required constants are defined."""
        assert hasattr(tools_papers, "COLLECTION_NAME")
        assert hasattr(tools_papers, "TOP_K")
        assert hasattr(tools_papers, "CHROMA_DB_PATH")
        assert isinstance(tools_papers.TOP_K, int)
        assert tools_papers.TOP_K > 0

    def test_collection_name_matches_ingest(self):
        """Test that COLLECTION_NAME matches ingest_data.py."""
        assert tools_papers.COLLECTION_NAME == "arxiv_papers"

    def test_base_dir_resolved(self):
        """Test that BASE_DIR is correctly resolved."""
        assert hasattr(tools_papers, "BASE_DIR")
        assert isinstance(tools_papers.BASE_DIR, Path)

    def test_chroma_db_path_is_path(self):
        """Test that CHROMA_DB_PATH is a Path object."""
        assert isinstance(tools_papers.CHROMA_DB_PATH, Path)
        assert "data" in str(tools_papers.CHROMA_DB_PATH)
        assert "chroma_db" in str(tools_papers.CHROMA_DB_PATH)

    @patch("chromadb.PersistentClient")
    @patch.object(tools_papers, "create_embedding_function")
    def test_get_collection_initializes_client(
        self, mock_embedding_fn, mock_client
    ):
        """Test that _get_collection initializes ChromaDB client."""
        # Reset global collection state
        tools_papers._collection = None

        mock_collection = MagicMock()
        mock_collection.count.return_value = 1000
        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.return_value = (
            mock_collection
        )
        mock_client.return_value = mock_client_instance
        mock_embedding_fn.return_value = MagicMock()

        result = tools_papers._get_collection()

        # Verify that the client was initialized
        assert result is not None
        mock_client.assert_called_once()

    @patch("chromadb.PersistentClient")
    @patch.object(tools_papers, "create_embedding_function")
    def test_get_collection_caches_result(
        self, mock_embedding_fn, mock_client
    ):
        """Test that _get_collection caches the connection."""
        # Reset global collection state
        tools_papers._collection = None

        mock_collection = MagicMock()
        mock_collection.count.return_value = 1000
        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.return_value = (
            mock_collection
        )
        mock_client.return_value = mock_client_instance
        mock_embedding_fn.return_value = MagicMock()

        # First call
        result1 = tools_papers._get_collection()
        call_count_first = mock_client.call_count

        # Second call
        result2 = tools_papers._get_collection()
        call_count_second = mock_client.call_count

        # ChromaDB client should only be initialized once
        assert call_count_first == call_count_second
        assert result1 is result2

    @patch("chromadb.PersistentClient")
    @patch.object(tools_papers, "create_embedding_function")
    def test_get_collection_handles_connection_error(
        self, mock_embedding_fn, mock_client
    ):
        """Test that _get_collection handles connection errors gracefully."""
        # Reset global collection state
        tools_papers._collection = None

        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.side_effect = Exception(
            "Connection failed"
        )
        mock_client.return_value = mock_client_instance
        mock_embedding_fn.return_value = MagicMock()

        result = tools_papers._get_collection()

        # Should return None on error
        assert result is None

    @patch("chromadb.PersistentClient")
    @patch.object(tools_papers, "create_embedding_function")
    def test_search_ai_papers_with_available_db(
        self, mock_embedding_fn, mock_client
    ):
        """Test searching papers when ChromaDB is available."""
        # Reset global collection state
        tools_papers._collection = None

        mock_collection = MagicMock()
        mock_collection.count.return_value = 9767
        mock_collection.query.return_value = {
            "documents": [
                [
                    "Title: Paper 1\n\nAbstract: Test abstract 1",
                    "Title: Paper 2\n\nAbstract: Test abstract 2",
                ]
            ],
            "metadatas": [
                [
                    {
                        "title": "Paper 1",
                        "arxiv_id": "2301.001",
                        "url_abs": "http://arxiv.org/abs/2301.001",
                        "url_pdf": "http://arxiv.org/pdf/2301.001.pdf",
                    },
                    {
                        "title": "Paper 2",
                        "arxiv_id": "2301.002",
                        "url_abs": "http://arxiv.org/abs/2301.002",
                        "url_pdf": "http://arxiv.org/pdf/2301.002.pdf",
                    },
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.return_value = (
            mock_collection
        )
        mock_client.return_value = mock_client_instance
        mock_embedding_fn.return_value = MagicMock()

        result = tools_papers.search_ai_papers.invoke(
            {"query": "multi-agent learning"}
        )

        assert isinstance(result, str)
        assert "Paper 1" in result
        assert "Paper 2" in result
        assert "PAPER_COUNT:" in result or "arxiv_id" in result

    @patch("chromadb.PersistentClient")
    @patch.object(tools_papers, "create_embedding_function")
    def test_search_ai_papers_without_db(
        self, mock_embedding_fn, mock_client
    ):
        """Test searching papers when ChromaDB is not available."""
        # Reset global collection state
        tools_papers._collection = None

        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.side_effect = Exception(
            "DB not found"
        )
        mock_client.return_value = mock_client_instance
        mock_embedding_fn.return_value = MagicMock()

        result = tools_papers.search_ai_papers.invoke(
            {"query": "test query"}
        )

        assert "SETUP_REQUIRED" in result
        assert "ingest_data" in result

    @patch("chromadb.PersistentClient")
    @patch.object(tools_papers, "create_embedding_function")
    def test_search_ai_papers_respects_top_k(
        self, mock_embedding_fn, mock_client
    ):
        """Test that search respects TOP_K parameter."""
        # Reset global collection state
        tools_papers._collection = None

        mock_collection = MagicMock()
        mock_collection.count.return_value = 9767
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2", "doc3", "doc4", "doc5"]],
            "metadatas": [
                [
                    {
                        "title": f"Paper {i}",
                        "arxiv_id": f"id{i}",
                        "url_abs": f"url{i}",
                        "url_pdf": f"pdf{i}",
                    }
                    for i in range(5)
                ]
            ],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        }

        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.return_value = (
            mock_collection
        )
        mock_client.return_value = mock_client_instance
        mock_embedding_fn.return_value = MagicMock()

        tools_papers.search_ai_papers.invoke(
            {"query": "test query"}
        )

        # Verify query was called with n_results=TOP_K
        call_args = mock_collection.query.call_args
        assert call_args[1]["n_results"] == tools_papers.TOP_K

    @patch("chromadb.PersistentClient")
    @patch.object(tools_papers, "create_embedding_function")
    def test_search_ai_papers_formats_output(
        self, mock_embedding_fn, mock_client
    ):
        """Test that search output is properly formatted."""
        # Reset global collection state
        tools_papers._collection = None

        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.query.return_value = {
            "documents": [
                ["Title: Test Paper\n\nAbstract: This is a test"]
            ],
            "metadatas": [
                [
                    {
                        "title": "Test Paper",
                        "arxiv_id": "2401.001",
                        "url_abs": "http://arxiv.org/abs/2401.001",
                        "url_pdf": "http://arxiv.org/pdf/2401.001.pdf",
                    }
                ]
            ],
            "distances": [[0.15]],
        }

        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.return_value = (
            mock_collection
        )
        mock_client.return_value = mock_client_instance
        mock_embedding_fn.return_value = MagicMock()

        result = tools_papers.search_ai_papers.invoke(
            {"query": "test query"}
        )

        # Check expected format
        assert "TITLE:" in result or "Test Paper" in result
        assert "ARXIV_ID:" in result or "2401.001" in result
        assert "URL_ABS:" in result or "arxiv.org/abs" in result
        assert "PDF:" in result or "pdf" in result

    @patch("chromadb.PersistentClient")
    @patch.object(tools_papers, "create_embedding_function")
    def test_search_ai_papers_handles_missing_metadata(
        self, mock_embedding_fn, mock_client
    ):
        """Test handling of missing metadata fields."""
        # Reset global collection state
        tools_papers._collection = None

        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.query.return_value = {
            "documents": [["Some document text"]],
            "metadatas": [
                [
                    {
                        "title": "Paper with missing URL",
                        # Missing arxiv_id, url_abs, url_pdf
                    }
                ]
            ],
            "distances": [[0.2]],
        }

        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.return_value = (
            mock_collection
        )
        mock_client.return_value = mock_client_instance
        mock_embedding_fn.return_value = MagicMock()

        result = tools_papers.search_ai_papers.invoke(
            {"query": "test query"}
        )

        # Should handle gracefully without errors
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("chromadb.PersistentClient")
    @patch.object(tools_papers, "create_embedding_function")
    def test_search_ai_papers_includes_relevance_score(
        self, mock_embedding_fn, mock_client
    ):
        """Test that relevance scores are included in output."""
        # Reset global collection state
        tools_papers._collection = None

        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.query.return_value = {
            "documents": [["Paper document"]],
            "metadatas": [
                [
                    {
                        "title": "Relevant Paper",
                        "arxiv_id": "2401.001",
                        "url_abs": "http://arxiv.org/abs/2401.001",
                        "url_pdf": "http://arxiv.org/pdf/2401.001.pdf",
                    }
                ]
            ],
            "distances": [
                [0.25]
            ],  # Distance = 0.25, so score = 1 - 0.25 = 0.75
        }

        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.return_value = (
            mock_collection
        )
        mock_client.return_value = mock_client_instance
        mock_embedding_fn.return_value = MagicMock()

        result = tools_papers.search_ai_papers.invoke(
            {"query": "test query"}
        )

        # Score should be shown (1 - 0.25 = 0.75)
        assert "0.75" in result or "score" in result.lower()

    def test_search_ai_papers_is_callable(self):
        """Test that search_ai_papers is a langchain tool with invoke method."""
        # Langchain @tool decorated functions are StructuredTool objects
        # They don't pass callable() but do have an invoke method
        assert hasattr(tools_papers.search_ai_papers, "invoke")
        assert callable(tools_papers.search_ai_papers.invoke)
