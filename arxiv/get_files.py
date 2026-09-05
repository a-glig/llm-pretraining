import csv
import requests
import tarfile
from pathlib import Path
import time

def download_papers(csv_path = "arxiv/papers.csv", output_dir = "arxiv/raw"):
    """Download papers from arxiv with identifiers specified in a .csv file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline = "", encoding = "utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            arxiv_id = row["id"]

            try:
                url = f"https://export.arxiv.org/e-print/{arxiv_id}"
                response = requests.get(url)
                response.raise_for_status()

                archive = output_dir / f"{arxiv_id}.tar"
                archive.write_bytes(response.content)

                print(f"Downloaded {arxiv_id}.")

            except Exception as e:
                print(f"Failed to download {arxiv_id}: {e}.")

            time.sleep(3)


def extract_papers(csv_path = "arxiv/papers.csv", raw_dir = "arxiv/raw",
                   output_dir = "arxiv/extracted"):
    """Extract all .tar files and save files in separate folder."""
    raw_dir = Path("arxiv/raw")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    with open(csv_path, newline = "", encoding = "utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            arxiv_id = row["id"]

            try:
                archive = raw_dir / f"{arxiv_id}.tar"
                paper_dir = output_dir / arxiv_id
                paper_dir.mkdir(parents = True, exist_ok = True)

                with tarfile.open(archive) as tar:
                    tar.extractall(paper_dir)

                print(f"Extracted {arxiv_id}.")

            except Exception as e:
                print(f"Failed to extract {arxiv_id}: {e}.")


# Extract and download papers
download_papers()
extract_papers()