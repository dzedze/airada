import re
import defusedxml.ElementTree as ET
from datetime import datetime

import requests
from langchain.tools import tool

# RSS feeds to pull from
RSS_FEEDS = {
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "MIT Tech Review": "https://www.technologyreview.com/feed/",
    "The Decoder": "https://the-decoder.com/feed/",
}

MAX_ITEMS_PER_FEED = 6


def _clean_html(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_date(raw: str) -> str:
    """Normalise RSS date strings to YYYY-MM-DD."""
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime(
                "%Y-%m-%d"
            )
        except ValueError:
            continue
    return raw.strip()


def _fetch_feed(feed_url: str, source_name: str) -> list:
    """Fetch and parse one RSS feed. Returns a list of article dicts."""
    resp = requests.get(
        feed_url,
        timeout=10,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; AITrendAgent/1.0; "
                "+https://github.com/ai-trend-agent)"
            )
        },
    )
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    channel = root.find("channel")
    if channel is None:
        channel = root
    items = channel.findall("item")[:MAX_ITEMS_PER_FEED]

    articles = []
    for item in items:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        desc = (item.findtext("description") or "").strip()

        if not title or not link:
            continue

        articles.append(
            {
                "title": title,
                "url": link,
                "summary": _clean_html(desc)[:400],
                "published": _parse_date(pub) if pub else "",
                "source": source_name,
            }
        )

    return articles


@tool
def get_ai_news(query: str) -> str:
    """
    Fetch the latest AI/LLM news from live RSS feeds and return a raw digest.

    ONLY use this tool when the user asks about current events, recent announcements,
    industry news, product launches, or what's trending in AI right now.
    Do NOT use for GitHub repos or research papers.

    Examples of queries that should trigger this tool:
    - "What's new in AI this week?"
    - "Latest LLM announcements"
    - "What did OpenAI release recently?"
    - "Summarize today's AI news"
    - "What's happening in AI right now?"

    Returns a raw structured digest of articles (title, url, date, source, summary).
    The calling LLM is responsible for formatting the final answer.
    """
    print(f"[News] Fetching RSS feeds for: '{query}'")

    all_articles = []
    errors = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            articles = _fetch_feed(feed_url, source_name)
            all_articles.extend(articles)
            print(f"[News] {source_name}: {len(articles)} articles")
        except requests.exceptions.HTTPError as e:
            msg = f"{source_name}: HTTP {e.response.status_code}"
            errors.append(msg)
            print(f"[News]   {msg}")
        except Exception as e:
            msg = f"{source_name}: {e}"
            errors.append(msg)
            print(f"[News]   {msg}")

    if not all_articles:
        error_detail = (
            "; ".join(errors) if errors else "unknown error"
        )
        return f"FETCH_FAILED: Could not retrieve articles from any feed. Errors: {error_detail}"

    # Sort newest first
    all_articles.sort(
        key=lambda a: a.get("published") or "", reverse=True
    )

    # Build a plain structured digest
    lines = [f"ARTICLE_COUNT: {len(all_articles)}", ""]
    for i, a in enumerate(all_articles, 1):
        lines.append(f"[{i}]")
        lines.append(f"TITLE:     {a['title']}")
        lines.append(f"URL:       {a['url']}")
        lines.append(f"SOURCE:    {a['source']}")
        lines.append(f"DATE:      {a['published'] or 'N/A'}")
        lines.append(f"SUMMARY:   {a['summary']}")
        lines.append("")

    if errors:
        lines.append(f"PARTIAL_ERRORS: {'; '.join(errors)}")

    return "\n".join(lines)


# --- Testing ---
if __name__ == "__main__":
    print(
        get_ai_news.invoke({"query": "What's new in AI this week?"})
    )
