import re

from arxiv.find_main import find_main_tex

# Example: Swampland review
arxiv_id = "2102.01111"
main_file = find_main_tex(arxiv_id)

def get_raw_text(tex_file):
    """Load main .tex file and extract raw text."""
    with open(tex_file, "r", encoding = "utf-8") as file:
        return file.read()


def remove_preamble_postamble(text):
    """Keep only text within document environment."""
    start = text.find(r"\begin{document}")
    start += len(r"\begin{document}")
    end = text.find(r"\end{document}")

    return text[start:end]

raw_text = get_raw_text(main_file)
text = remove_preamble_postamble(raw_text)
print(text[:4000])