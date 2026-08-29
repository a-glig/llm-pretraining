import csv
import requests
from pathlib import Path
import time

raw_dir = Path("arxiv/raw")
raw_dir.mkdir(parents = True, exist_ok = True)

with open("arxiv/papers.csv", newline = "", encoding= "utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        arxiv_id = row["id"]

        try:
            url = f"https://export.arxiv.org/e-print/{arxiv_id}"
            response = requests.get(url)
            response.raise_for_status()

            archive = raw_dir / f"{arxiv_id}.tar"
            archive.write_bytes(response.content)

            print(f"Downloaded {arxiv_id}.")

        except Exception as e:
            print(f"Failed to download {arxiv_id}: {e}.")

        time.sleep(3)