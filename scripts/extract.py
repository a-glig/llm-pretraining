import csv
import tarfile
from pathlib import Path

raw_dir = Path("arxiv/raw")
extracted_dir = Path("arxiv/extracted")
extracted_dir.mkdir(parents = True, exist_ok = True)

with open("data/papers.csv", newline = "", encoding = "utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        arxiv_id = row["id"]

        try:
            archive = raw_dir / f"{arxiv_id}.tar"
            output_dir = extracted_dir / arxiv_id
            output_dir.mkdir(parents = True, exist_ok = True)

            with tarfile.open(archive) as tar:
                tar.extractall(output_dir)

            print(f"Extracted {arxiv_id}.")

        except Exception as e:
            print(f"Failed to extract {arxiv_id}: {e}.")

