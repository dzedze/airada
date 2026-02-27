"""
One-time setup script — run this BEFORE starting the app.

Reads the papers CSV, embeds every title+abstract with OpenAI embeddings,
and persists the vector index to ChromaDB.

Usage:
    python ingest_data.py

Upsert semantics: safe to re-run — no duplicate entries will be created.
"""
import os
import chromadb
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.providers.llm_factory import create_embedding_function

# --- Configuration (must match tools_papers.py) ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = BASE_DIR / "data" / "processed" / "papers_agents_llm_subset.csv"
CHROMA_DB_PATH  = BASE_DIR / "data" / "vector" / "chroma_db"
COLLECTION_NAME = "arxiv_papers"
BATCH_SIZE = 100

def build_document(row: pd.Series) -> str:
    """Combine title + abstract into one text chunk for embedding."""
    title = str(row.get("title",    "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    return f"Title: {title}\n\nAbstract: {abstract}"

def build_metadata(row: pd.Series) -> dict:
    """Store useful fields as ChromaDB metadata for later retrieval."""
    return {
        "title": str(row.get("title",    "")).strip(),
        "arxiv_id": str(row.get("arxiv_id", "")).strip(),
        "url_abs": str(row.get("url_abs",  "")).strip(),
        "url_pdf": str(row.get("url_pdf",  "")).strip(),
    }

def ingest():
    # ── Load CSV ──────────────────────────────────────────────
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"{len(df):,} rows, columns: {list(df.columns)}\n")

    # make sure the data have "title" and "abstract" columns, 
    # those contain relevant information for the tool
    required = {"title", "abstract"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.dropna(subset=["title", "abstract"], how="all").reset_index(drop=True)
    print(f"{len(df):,} papers after dropping empty rows\n")

    # ── Connect to ChromaDB ───────────────────────────────────
    print(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")

    ef = create_embedding_function()
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection '{COLLECTION_NAME}' ready "
          f"({collection.count():,} docs already stored)\n")

    # ── Build document list ───────────────────────────────────
    documents, metadatas, ids = [], [], []
    for _, row in df.iterrows():
        doc = build_document(row)
        if not doc.strip():
            continue
        documents.append(doc)
        metadatas.append(build_metadata(row))
        ids.append(str(row["arxiv_id"]))

    # ── Upsert in batches ─────────────────────────────────────
    total = len(documents)
    print(f"Embedding & inserting {total:,} papers ")

    for start in tqdm(range(0, total, BATCH_SIZE), desc="Ingesting", unit="batch"):
        end = min(start + BATCH_SIZE, total)
        collection.upsert(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    print(f"\nDone! '{COLLECTION_NAME}' now holds {collection.count():,} papers.")
    print(f"Persisted at: {os.path.abspath(CHROMA_DB_PATH)}\n")
    print("You can now run the app:  python app.py")


if __name__ == "__main__":
    ingest()
