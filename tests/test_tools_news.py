"""
Tests for src/tools/tools_news.py
"""

import pytest
from unittest.mock import patch, MagicMock
import requests_mock
from datetime import datetime

from importlib import import_module
tools_news = import_module('src.tools.tools_news')


class TestNewsTools:
    """Test cases for AI news fetching tool."""

    def test_rss_feeds_defined(self):
        """Test that RSS_FEEDS dictionary is defined."""
        assert hasattr(tools_news, 'RSS_FEEDS')
        assert isinstance(tools_news.RSS_FEEDS, dict)
        assert len(tools_news.RSS_FEEDS) > 0

    def test_max_items_per_feed_defined(self):
        """Test that MAX_ITEMS_PER_FEED constant is set."""
        assert hasattr(tools_news, 'MAX_ITEMS_PER_FEED')
        assert isinstance(tools_news.MAX_ITEMS_PER_FEED, int)
        assert tools_news.MAX_ITEMS_PER_FEED > 0

    def test_clean_html_removes_tags(self):
        """Test that _clean_html removes HTML tags."""
        html_text = "<p>Hello <b>world</b></p>"
        result = tools_news._clean_html(html_text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_clean_html_collapses_whitespace(self):
        """Test that _clean_html collapses multiple spaces."""
        text = "Hello    world   test"
        result = tools_news._clean_html(text)
        assert "    " not in result
        assert result == "Hello world test"

    def test_parse_date_rfc2822_format(self):
        """Test parsing RFC 2822 date format."""
        date_str = "Mon, 01 Apr 2024 10:30:00 +0000"
        result = tools_news._parse_date(date_str)
        assert "2024-04-01" in result

    def test_parse_date_iso_format(self):
        """Test parsing ISO 8601 date format."""
        date_str = "2024-04-01T10:30:00+0000"
        result = tools_news._parse_date(date_str)
        assert "2024-04-01" in result

    def test_parse_date_invalid_format(self):
        """Test parsing invalid date format returns raw string."""
        date_str = "invalid-date"
        result = tools_news._parse_date(date_str)
        assert result == "invalid-date"

    def test_fetch_feed_success(self):
        """Test successfully fetching and parsing RSS feed."""
        rss_xml = """<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Test Article 1</title>
                    <link>https://example.com/article1</link>
                    <pubDate>Mon, 01 Apr 2024 10:00:00 +0000</pubDate>
                    <description><![CDATA[This is a test article]]></description>
                </item>
                <item>
                    <title>Test Article 2</title>
                    <link>https://example.com/article2</link>
                    <pubDate>Mon, 01 Apr 2024 11:00:00 +0000</pubDate>
                    <description><![CDATA[Another test article]]></description>
                </item>
            </channel>
        </rss>"""

        with requests_mock.Mocker() as m:
            m.get("https://example.com/feed", text=rss_xml)
            articles = tools_news._fetch_feed("https://example.com/feed", "Test Feed")

            assert len(articles) == 2
            assert articles[0]["title"] == "Test Article 1"
            assert articles[0]["url"] == "https://example.com/article1"
            assert articles[0]["source"] == "Test Feed"
            assert "2024-04-01" in articles[0]["published"]

    def test_fetch_feed_respects_max_items(self):
        """Test that feed parsing respects MAX_ITEMS_PER_FEED."""
        rss_xml = """<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Article 1</title>
                    <link>https://example.com/1</link>
                    <description>Desc 1</description>
                </item>
                <item>
                    <title>Article 2</title>
                    <link>https://example.com/2</link>
                    <description>Desc 2</description>
                </item>
                <item>
                    <title>Article 3</title>
                    <link>https://example.com/3</link>
                    <description>Desc 3</description>
                </item>
                <item>
                    <title>Article 4</title>
                    <link>https://example.com/4</link>
                    <description>Desc 4</description>
                </item>
                <item>
                    <title>Article 5</title>
                    <link>https://example.com/5</link>
                    <description>Desc 5</description>
                </item>
            </channel>
        </rss>"""

        with requests_mock.Mocker() as m:
            m.get("https://example.com/feed", text=rss_xml)
            articles = tools_news._fetch_feed("https://example.com/feed", "Test")

            # Should be limited by MAX_ITEMS_PER_FEED
            assert len(articles) <= tools_news.MAX_ITEMS_PER_FEED

    def test_fetch_feed_skips_incomplete_items(self):
        """Test that items without title or link are skipped."""
        rss_xml = """<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Valid Article</title>
                    <link>https://example.com/valid</link>
                    <description>Valid</description>
                </item>
                <item>
                    <title>Missing Link</title>
                    <description>No link provided</description>
                </item>
                <item>
                    <link>https://example.com/notitle</link>
                    <description>No title</description>
                </item>
            </channel>
        </rss>"""

        with requests_mock.Mocker() as m:
            m.get("https://example.com/feed", text=rss_xml)
            articles = tools_news._fetch_feed("https://example.com/feed", "Test")

            # Only the complete item should be parsed
            assert len(articles) == 1
            assert articles[0]["title"] == "Valid Article"

    def test_fetch_feed_http_error(self):
        """Test HTTP error handling in feed fetch."""
        with requests_mock.Mocker() as m:
            m.get("https://example.com/feed", status_code=404)

            with pytest.raises(Exception):
                tools_news._fetch_feed("https://example.com/feed", "Test")

    def test_fetch_feed_invalid_xml(self):
        """Test handling of invalid XML in feed."""
        with requests_mock.Mocker() as m:
            m.get("https://example.com/feed", text="Invalid XML content <not> closed")

            with pytest.raises(Exception):
                tools_news._fetch_feed("https://example.com/feed", "Test")

    def test_get_ai_news_fetches_all_feeds(self):
        """Test that get_ai_news fetches from all configured feeds."""
        rss_xml = """<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Test Article</title>
                    <link>https://example.com/article</link>
                    <description>Test</description>
                </item>
            </channel>
        </rss>"""

        with requests_mock.Mocker() as m:
            # Mock all configured RSS feeds
            for feed_url in tools_news.RSS_FEEDS.values():
                m.get(feed_url, text=rss_xml)

            result = tools_news.get_ai_news.invoke({"query": "What's new in AI?"})

            assert isinstance(result, str)
            assert "ARTICLE_COUNT" in result
            assert "Test Article" in result

    def test_get_ai_news_partial_feed_failure(self):
        """Test handling when some feeds fail but others succeed."""
        valid_rss = """<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Working Article</title>
                    <link>https://example.com/article</link>
                    <description>From working feed</description>
                </item>
            </channel>
        </rss>"""

        with requests_mock.Mocker() as m:
            feed_urls = list(tools_news.RSS_FEEDS.values())
            # First feed works
            m.get(feed_urls[0], text=valid_rss)
            # Other feeds fail
            for feed_url in feed_urls[1:]:
                m.get(feed_url, status_code=500)

            result = tools_news.get_ai_news.invoke({"query": "AI news"})

            assert isinstance(result, str)
            assert "Working Article" in result

    def test_get_ai_news_all_feeds_fail(self):
        """Test handling when all feeds fail."""
        with requests_mock.Mocker() as m:
            for feed_url in tools_news.RSS_FEEDS.values():
                m.get(feed_url, status_code=500)

            result = tools_news.get_ai_news.invoke({"query": "AI news"})

            assert "FETCH_FAILED" in result

    def test_get_ai_news_sorts_by_date(self):
        """Test that articles are sorted by published date (newest first)."""
        rss_old = """<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Old Article from Feed1</title>
                    <link>https://example.com/old</link>
                    <pubDate>Mon, 01 Apr 2024 08:00:00 +0000</pubDate>
                    <description>Old</description>
                </item>
            </channel>
        </rss>"""

        rss_new = """<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>New Article from Feed2</title>
                    <link>https://example.com/new</link>
                    <pubDate>Tue, 02 Apr 2024 12:00:00 +0000</pubDate>
                    <description>New</description>
                </item>
            </channel>
        </rss>"""

        with requests_mock.Mocker() as m:
            feed_urls = list(tools_news.RSS_FEEDS.values())
            m.get(feed_urls[0], text=rss_old)
            m.get(feed_urls[1], text=rss_new)
            for feed_url in feed_urls[2:]:
                m.get(feed_url, status_code=500)

            result = tools_news.get_ai_news.invoke({"query": "AI news"})

            # "New Article" (2024-04-02) should appear before "Old Article" (2024-04-01)
            # when sorted by date descending (newest first)
            new_pos = result.find("New Article")
            old_pos = result.find("Old Article")
            # Both should be found
            assert new_pos >= 0 and old_pos >= 0, "Both articles should be in output"
            # Newest (04-02) should come first
            assert new_pos < old_pos, "Newer article should appear before older article"

    def test_get_ai_news_formats_output(self):
        """Test that output is properly formatted."""
        rss_xml = """<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Test Article</title>
                    <link>https://example.com/article</link>
                    <pubDate>Mon, 01 Apr 2024 10:00:00 +0000</pubDate>
                    <description>Test description</description>
                </item>
            </channel>
        </rss>"""

        with requests_mock.Mocker() as m:
            for feed_url in tools_news.RSS_FEEDS.values():
                m.get(feed_url, text=rss_xml)

            result = tools_news.get_ai_news.invoke({"query": "AI news"})

            # Check expected format
            assert "ARTICLE_COUNT:" in result
            assert "[1]" in result or "[" in result
            assert "TITLE:" in result
            assert "URL:" in result
            assert "SOURCE:" in result
            assert "DATE:" in result
            assert "SUMMARY:" in result
