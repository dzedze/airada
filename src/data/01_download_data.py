import requests
from pathlib import Path
from tqdm import tqdm

# data url
url = "https://huggingface.co/datasets/J0nasW/paperswithcode/resolve/main/papers.csv"

# Save path
save_path = (
    Path(__file__).parent.parent.parent
    / "data"
    / "raw"
    / "papers.csv"
)
save_path.parent.mkdir(parents=True, exist_ok=True)


def download_data():
    # Download
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with (
        open(save_path, "wb") as f,
        tqdm(
            desc="Downloading papers.csv",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"Downloaded to {save_path}")


if __name__ == "__main__":
    download_data()
