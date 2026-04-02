import chromadb
from langchain.tools import tool
from src.providers.llm_factory import create_embedding_function
from pathlib import Path

# --- Configuration (must match ingest_data.py) ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_PATH = BASE_DIR / "data" / "vector" / "chroma_db"
COLLECTION_NAME = "arxiv_papers"
# Number of top relevant vectors to retrieve
TOP_K = 5

# --- Lazy ChromaDB connection ---

_collection = None


def _get_collection():
    """Connect to ChromaDB once and reuse the connection for all queries."""
    global _collection
    if _collection is None:
        ef = create_embedding_function()
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            _collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
            )
            print(
                f"Connected to '{COLLECTION_NAME}' ({_collection.count():,} papers)"
            )
        except Exception as e:
            print(f"Could not load ChromaDB collection: {e}")
            print(
                "Run `python ingest_data.py` first to build the vector index."
            )
            _collection = None
    return _collection


# --- The Tool ---
@tool
def search_ai_papers(query: str) -> str:
    """
    Semantic search 9767 arXiv research papers on AI, LLMs, and Agents.

    ONLY use this tool when the user asks about research papers, academic studies,
    model comparisons, research trends, or literature-based explanations.
    Do NOT use for GitHub repos or current news/events.

    Examples of queries that should trigger this tool:
    - "Compare GPT-3 and PaLM papers"
    - "Summarize top LLM research trends"
    - "What do papers say about chain-of-thought prompting?"
    - "Find research on multi-agent debate"

    Returns a raw structured context block with retrieved papers.
    The calling LLM is responsible for synthesising the final answer.
    """
    collection = _get_collection()
    if collection is None:
        return (
            "SETUP_REQUIRED: The paper database has not been built yet. "
            "Run `python ingest_data.py` once to embed the CSV into ChromaDB, "
            "then restart the app."
        )

    print(f"[Papers RAG] Querying: '{query}'")

    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    papers = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        papers.append(
            {
                "document": doc,
                "title": meta.get("title", "Unknown"),
                "arxiv_id": meta.get("arxiv_id", ""),
                "url_abs": meta.get("url_abs", ""),
                "url_pdf": meta.get("url_pdf", ""),
                "score": round(1 - dist, 4),
            }
        )

    # Return a raw labelled context block — no LLM here
    lines = [f"PAPER_COUNT: {len(papers)}", ""]
    for i, p in enumerate(papers, 1):
        lines.append(f"[{i}]")
        lines.append(f"TITLE:    {p['title']}")
        lines.append(f"ARXIV_ID: {p['arxiv_id']}")
        lines.append(f"URL_ABS:  {p['url_abs']}")
        lines.append(f"URL_PDF:  {p['url_pdf']}")
        lines.append(f"SCORE:    {p['score']}")
        lines.append(f"CONTENT:  {p['document']}")
        lines.append("")

    return "\n".join(lines)


# --- Testing ---
if __name__ == "__main__":
    print(
        search_ai_papers.invoke(
            {"query": "Summarize top LLM research trends"}
        )
    )
