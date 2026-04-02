"""
Tests for src/data/01_download_data.py
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import tempfile

# Import the download_data module directly
from importlib import import_module
dl = import_module('src.data.01_download_data')


class TestDownloadData:
    """Test cases for download_data module."""

    def test_save_path_configured(self):
        """Test that save_path is properly configured."""
        assert dl.save_path is not None
        assert isinstance(dl.save_path, Path)
        assert "papers.csv" in str(dl.save_path)

    def test_url_is_valid(self):
        """Test that URL is defined and valid."""
        assert dl.url is not None
        assert isinstance(dl.url, str)
        assert dl.url.startswith("http")
        assert "papers.csv" in dl.url or "dataset" in dl.url

    @patch('requests.get')
    def test_download_data_makes_request(self, mock_get):
        """Test that download_data makes an HTTP request."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content = MagicMock(return_value=[b"chunk1", b"chunk2"])
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with patch('builtins.open', mock_open()):
            dl.download_data()

        # Verify request was made
        mock_get.assert_called_once()
        assert mock_get.call_args[0][0] == dl.url

    @patch('requests.get')
    def test_download_data_handles_chunked_response(self, mock_get):
        """Test that download_data correctly handles chunked response."""
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(b"".join(chunks)))}
        mock_response.iter_content = MagicMock(return_value=chunks)
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        m = mock_open()
        with patch('builtins.open', m):
            dl.download_data()

        # Verify file was opened in write-binary mode
        m.assert_called_once()
        call_args = m.call_args
        assert call_args[0][1] == "wb"

    @patch('requests.get')
    def test_download_data_raises_on_http_error(self, mock_get):
        """Test that download_data raises on HTTP error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404")
        mock_get.return_value = mock_response

        with pytest.raises(Exception):
            dl.download_data()

    @patch('requests.get')
    def test_download_data_with_timeout(self, mock_get):
        """Test that download_data uses timeout parameter."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content = MagicMock(return_value=[])
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with patch('builtins.open', mock_open()):
            dl.download_data()

        # Verify timeout is set (120 seconds based on code)
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get('timeout') == 120

    @patch('requests.get')
    def test_download_data_uses_stream(self, mock_get):
        """Test that download_data uses streaming."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content = MagicMock(return_value=[])
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with patch('builtins.open', mock_open()):
            dl.download_data()

        # Verify stream=True is used
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get('stream') is True
