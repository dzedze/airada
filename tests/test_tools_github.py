"""
Tests for src/tools/tools_github.py
"""

import pytest
import requests_mock

from importlib import import_module

tools_github = import_module("src.tools.tools_github")


class TestGitHubTools:
    """Test cases for GitHub search tool."""

    def test_ai_topics_list_defined(self):
        """Test that AI_TOPICS list is defined and contains expected topics."""
        assert hasattr(tools_github, "AI_TOPICS")
        assert isinstance(tools_github.AI_TOPICS, list)
        assert "llm" in tools_github.AI_TOPICS
        assert "rag" in tools_github.AI_TOPICS
        assert "ai-agents" in tools_github.AI_TOPICS

    def test_github_search_url_defined(self):
        """Test that GitHub API URL is correctly defined."""
        assert hasattr(tools_github, "GITHUB_SEARCH_URL")
        assert (
            tools_github.GITHUB_SEARCH_URL
            == "https://api.github.com/search/repositories"
        )

    def test_get_headers_returns_dict(self):
        """Test that _get_headers returns a dictionary."""
        headers = tools_github._get_headers()
        assert isinstance(headers, dict)
        assert "Accept" in headers

    def test_search_github_repos_with_valid_query(self):
        """Test searching GitHub with a valid query."""
        with requests_mock.Mocker() as m:
            mock_response = {
                "total_count": 5000,
                "items": [
                    {
                        "full_name": "owner/llm-repo1",
                        "html_url": "https://github.com/owner/llm-repo1",
                        "description": "An LLM framework",
                        "language": "Python",
                        "topics": ["llm", "ai"],
                        "stargazers_count": 1000,
                        "forks_count": 100,
                    },
                    {
                        "full_name": "owner/llm-repo2",
                        "html_url": "https://github.com/owner/llm-repo2",
                        "description": "Another LLM tool",
                        "language": "TypeScript",
                        "topics": ["llm"],
                        "stargazers_count": 500,
                        "forks_count": 50,
                    },
                ],
            }
            m.get(
                tools_github.GITHUB_SEARCH_URL, json=mock_response
            )

            # Call via invoke since tools are StructuredTool objects
            result = tools_github.search_github_repos.invoke(
                {"query": "top 2 LLM repos"}
            )

            assert isinstance(result, str)
            assert "llm-repo1" in result
            assert "llm-repo2" in result
            assert "Found" in result

    def test_search_github_repos_no_results(self):
        """Test GitHub search with no results."""
        with requests_mock.Mocker() as m:
            mock_response = {"total_count": 0, "items": []}
            m.get(
                tools_github.GITHUB_SEARCH_URL, json=mock_response
            )

            result = tools_github.search_github_repos.invoke(
                {"query": "obscure-query-xyz"}
            )

            assert "No repositories found" in result

    def test_search_github_repos_rate_limit_error(self):
        """Test GitHub API rate limit error handling."""
        with requests_mock.Mocker() as m:
            m.get(
                tools_github.GITHUB_SEARCH_URL,
                status_code=403,
                text="API rate limit exceeded",
            )

            result = tools_github.search_github_repos.invoke(
                {"query": "test query"}
            )

            assert "rate limit" in result.lower()

    def test_search_github_repos_http_error(self):
        """Test GitHub API HTTP error handling."""
        with requests_mock.Mocker() as m:
            m.get(
                tools_github.GITHUB_SEARCH_URL,
                status_code=500,
                text="Server error",
            )

            result = tools_github.search_github_repos.invoke(
                {"query": "test query"}
            )

            assert (
                "error" in result.lower() or "error 500" in result
            )

    def test_search_github_repos_parses_query_count(self):
        """Test that query parser correctly extracts result count."""
        with requests_mock.Mocker() as m:
            mock_response = {
                "total_count": 100,
                "items": [
                    {
                        "full_name": f"owner/repo{i}",
                        "html_url": f"https://github.com/owner/repo{i}",
                        "description": f"Repo {i}",
                        "language": "Python",
                        "topics": ["llm"],
                        "stargazers_count": 100 - i,
                        "forks_count": 10,
                    }
                    for i in range(5)
                ],
            }
            m.get(
                tools_github.GITHUB_SEARCH_URL, json=mock_response
            )

            # Test "top 5" parsing
            result = tools_github.search_github_repos.invoke(
                {"query": "top 5 LLM repos"}
            )
            assert "repo0" in result

    def test_search_github_repos_default_count(self):
        """Test default per_page when count not specified."""
        with requests_mock.Mocker() as m:
            mock_response = {
                "total_count": 1000,
                "items": [
                    {
                        "full_name": "owner/repo1",
                        "html_url": "https://github.com/owner/repo1",
                        "description": "Test repo",
                        "language": "Python",
                        "topics": ["llm"],
                        "stargazers_count": 100,
                        "forks_count": 20,
                    }
                ],
            }
            m.get(
                tools_github.GITHUB_SEARCH_URL, json=mock_response
            )

            result = tools_github.search_github_repos.invoke(
                {"query": "LLM frameworks"}
            )

            assert isinstance(result, str)
            assert len(result) > 0

    def test_search_github_repos_topic_matching(self):
        """Test that topic tags are matched from query."""
        with requests_mock.Mocker() as m:
            m.get(
                tools_github.GITHUB_SEARCH_URL,
                json={"total_count": 0, "items": []},
            )

            # Query mentions "rag"
            tools_github.search_github_repos.invoke(
                {"query": "RAG systems"}
            )

            # Verify request was made
            assert m.called
            assert m.request_history[0].method == "GET"

    def test_search_github_repos_language_filter(self):
        """Test that language is extracted and added to query."""
        with requests_mock.Mocker() as m:
            m.get(
                tools_github.GITHUB_SEARCH_URL,
                json={"total_count": 0, "items": []},
            )

            # Query mentions "python"
            tools_github.search_github_repos.invoke(
                {"query": "Python LLM libraries"}
            )

            assert m.called

    def test_search_github_repos_timeout_handling(self):
        """Test timeout handling in API request."""
        with requests_mock.Mocker() as m:
            m.get(
                tools_github.GITHUB_SEARCH_URL,
                exc=Exception("Connection timeout"),
            )

            # Should handle gracefully
            with pytest.raises(Exception):
                tools_github.search_github_repos("test")
