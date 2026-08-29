import csv
from pathlib import Path

def find_main_tex(arxiv_id):
    """Find the main .tex file containing the bulk text of a paper."""

    paper_dir = Path("arxiv/extracted") / arxiv_id
    tex_files = Path(paper_dir).rglob("*.tex")
    main_candidates = []

    for tex_file in tex_files:
        text = tex_file.read_text(encoding = "utf-8", errors = "ignore")

        if r"\documentclass" in text and r"\begin{document}" in text:
            main_candidates.append(tex_file)


    if len(main_candidates) == 0:
        raise FileNotFoundError(f"No main .tex file found in {paper_dir}.")

    if len(main_candidates) > 1:
        raise RuntimeError(f"Multiple possible main files found in {paper_dir}.")

    return main_candidates[0]


def print_main_files():
    """Print all main .tex files from the dataset."""

    with open("arxiv/papers.csv", newline = "", encoding = "utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            arxiv_id = row["id"]

            try:
                main_file = find_main_tex(arxiv_id)
                print(f"{arxiv_id}: {main_file.name}")
            except Exception as e:
                print(f"{arxiv_id} failed: {e}")

print_main_files()