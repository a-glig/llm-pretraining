import re

from arxiv.find_main import find_main_tex
from arxiv.latex_patterns import REMOVE_COMMANDS
from arxiv.latex_patterns import PRESERVE_CONTENT
from arxiv.latex_patterns import REMOVE_CONTENT
from arxiv.latex_patterns import REMOVE_ENVIRONMENTS

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


def remove_comments(text):
    """Remove comments from the text."""
    return re.sub(r'(?<!\\)%.*','', text)


def clean_latex_patterns(text):
    """Remove LaTeX code not needed for pre-training."""
    text = _remove_enviroments(text)
    text = _remove_content(text)
    text = _preserve_content(text)
    text = _remove_commands(text)

    return text

def _remove_enviroments(text):
    """Remove specific LaTeX environments."""
    for environment in sorted(REMOVE_ENVIRONMENTS, key = len, reverse = True):
        pattern = (
            r'\\begin\{' + re.escape(environment) + r'\}.*?'
            r'\\end\{' + re.escape(environment) + r'\}'
        )
        text = re.sub(pattern, '', text, flags = re.DOTALL)

    return text


def _remove_content(text):
    """Remove LaTeX command and wrapped content."""
    for command in sorted(REMOVE_CONTENT, key = len, reverse = True):
        pattern = re.escape(command) + r'\*?(?:\[[^]]*\])?\{([^}]*)\}'
        text = re.sub(pattern, '', text)

    return text


def _preserve_content(text):
    """Remove LaTeX command but preserve wrapped content."""
    for command in sorted(PRESERVE_CONTENT, key = len, reverse = True):
        pattern = re.escape(command) + r'\{([^}]*)\}'
        text = re.sub(pattern, r'\1', text)

    return text


def _remove_commands(text):
    """Remove LaTeX formatting commands."""
    for command in sorted(REMOVE_COMMANDS, key = len, reverse = True):
        pattern = re.escape(command)
        text = re.sub(pattern, '', text)

    return text


def clean_whitespace(text):
    """Perform whitespace cleanup."""
    # Replace multiple tabs/spaces with a single space.
    text = re.sub(r'[ \t]+', ' ', text)

    # Replace line breaks within a paragraph with a space.
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Collapse three or more empty lines into two.
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


raw_text = get_raw_text(main_file)
text = remove_preamble_postamble(raw_text)
text = remove_comments(text)
text = clean_latex_patterns(text)
text = clean_whitespace(text)

with open("text_cleaned.txt", "w", encoding = "utf-8") as file:
    file.write(text)