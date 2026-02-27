from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re

raw_path = Path(__file__).parent.parent.parent / "data" / "raw" / "papers.csv"
processed_path = Path(__file__).parent.parent.parent / "data" / "processed"
processed_path.mkdir(parents=True, exist_ok=True)

def process_data():

    print("Loading papers.csv...")
    df = pd.read_csv(raw_path)
    print(f"Total papers loaded: {len(df)}")

    keywords = [
        # Agentic
        "agent", "agentic", "multi-agent", "autonomous agent",
        "tool use", "tool-using", "planner", "reasoning agent",
        "react", "toolformer",
        # RAG
        "retrieval", "retrieval-augmented", "retrieval augmented",
        "rag", "knowledge retrieval", "memory-augmented",
        # LLM general
        "language model", "large language model", "llm",
        "transformer", "gpt", "bert", "palm", "llama"
    ]

    # Use (?:...) non-capturing group to avoid the warning
    # pattern = r"(?i)(?:" + "|".join(re.escape(k) for k in keywords) + r")"
    pattern = r"(?i)\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b"
    steps = ["Filtering titles", "Filtering abstracts", "Selecting columns", "Saving file"]

    with tqdm(total=len(steps), desc="Processing", unit="step") as bar:
        bar.set_description(steps[0])
        title_mask = df["title"].str.contains(pattern, na=False, regex=True)
        bar.update(1)

        bar.set_description(steps[1])
        abstract_mask = df["abstract"].str.contains(pattern, na=False, regex=True)
        filtered = df[title_mask | abstract_mask]
        print(f"\nMatched papers: {len(filtered)}")
        bar.update(1)

        bar.set_description(steps[2])
        filtered = filtered[
            ["title", "abstract", "url_abs", "url_pdf", "arxiv_id"]
        ].dropna()
        print(f"Papers after dropping nulls: {len(filtered)}")
        bar.update(1)

        bar.set_description(steps[3])
        save_path = processed_path / "papers_agents_llm_subset.csv"
        filtered.to_csv(save_path, index=False)
        bar.update(1)

    print(f"Saved processed file to {save_path}")

if __name__ == "__main__":
    process_data()