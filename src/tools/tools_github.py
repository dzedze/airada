import re
import requests
from langchain.tools import tool

# import os
# from pathlib import Path
# from dotenv import load_dotenv

# # Load the .secrets file
# env_path = Path(__file__).parent.parent / ".secrets"
# load_dotenv(dotenv_path=env_path)

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"

# Supported topic tags
AI_TOPICS = [
    "ai-agents",
    "agentic-ai",
    "llm",
    "rag",
    "llm-inference",
    "vector-database",
    "prompt-engineering",
    "langchain",
    "openai",
    "huggingface",
    "machine-learning",
]


def _get_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    # If you have GITHUB_TOKEN put yours in the .secrets or .env
    # and load it accordingly to have more requests and tokens limit
    # token = os.getenv("GITHUB_TOKEN")
    # if token:
    #     headers["Authorization"] = f"Bearer {token}"
    return headers


@tool
def search_github_repos(query: str) -> str:
    """
    Find trending GitHub repositories on AI, LLM, agents, RAG, and related topics.

    ONLY use this tool when the user asks about GitHub repos, projects, libraries,
    frameworks, or open-source code. Do NOT use for news or research papers.

    Examples of queries that should trigger this tool:
    - "Show me the top 5 AI/LLM GitHub projects"
    - "Give me repos with the topic agentic-ai"
    - "Trending Python LLM frameworks on GitHub"
    - "Most starred RAG libraries"

    Pass the user's request as-is — topic tags and result count are parsed from it.
    """
    print(f"[GitHub] Searching for: '{query}'")

    query_lower = query.lower()

    # --- Parse how many results the user wants, e.g. "top 5" ---
    count_match = re.search(
        r"\btop\s*(\d+)\b|\b(\d+)\s*(repos?|projects?|results?)\b",
        query_lower,
    )
    if count_match:
        per_page = int(
            next(
                g for g in count_match.groups() if g and g.isdigit()
            )
        )
    else:
        per_page = 8
    per_page = max(1, min(per_page, 20))

    # --- Build GitHub search query from topic tags ---
    parts = []
    matched_topics = [
        t
        for t in AI_TOPICS
        if t.replace("-", " ") in query_lower or t in query_lower
    ]

    if matched_topics:
        parts.extend(f"topic:{t}" for t in matched_topics[:3])
    else:
        parts = ["topic:ai-agents", "topic:llm", "topic:agentic-ai"]

    # Extra plain keywords
    for kw in [
        "autonomous",
        "agent",
        "multiagent",
        "framework",
        "inference",
        "rag",
    ]:
        if kw in query_lower and f"topic:{kw}" not in " ".join(
            parts
        ):
            parts.append(kw)

    # Language filter
    language = None
    for lang in [
        "python",
        "typescript",
        "javascript",
        "rust",
        "go",
    ]:
        if lang in query_lower:
            language = lang
            break
    if language:
        parts.append(f"language:{language}")

    params = {
        "q": " ".join(parts),
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
    }

    try:
        response = requests.get(
            GITHUB_SEARCH_URL,
            headers=_get_headers(),
            params=params,
            timeout=10,
        )

        if response.status_code == 403:
            return "GitHub rate limit hit. Add a GITHUB_TOKEN to your .env for 5,000 req/hr."
        if not response.ok:
            return f"GitHub API error {response.status_code}: {response.text[:200]}"

        data = response.json()
        items = data.get("items", [])

        if not items:
            return f"No repositories found for query: {params['q']}"

        output = f"Found {data['total_count']:,} total repos. Showing top {len(items)}:\n\n"
        for i, repo in enumerate(items, 1):
            name = repo["full_name"]
            url = repo["html_url"]
            desc = repo.get("description") or "No description"
            lang = repo.get("language") or "N/A"
            topics = ", ".join(repo.get("topics", [])) or "N/A"
            stars = f"{repo['stargazers_count']:,}"
            forks = f"{repo['forks_count']:,}"

            # Clickable markdown link
            output += f"{i}. [{name}]({url}) ⭐ {stars}\n"
            output += f"   {desc}\n"
            output += f"   Language: {lang} · Forks: {forks}\n"
            output += f"   Topics: {topics}\n\n"

        return output

    except requests.exceptions.Timeout:
        return "GitHub request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Network error: {e}"


# --- Testing ---
if __name__ == "__main__":
    print(
        search_github_repos.invoke(
            {"query": "Show me the top 5 AI/LLM GitHub projects"}
        )
    )
    print("-" * 40)
    print(
        search_github_repos.invoke(
            {"query": "Give me repos with the topic agentic-ai"}
        )
    )
