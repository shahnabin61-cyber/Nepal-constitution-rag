import re
from pypdf import PdfReader


def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text


def chunk_by_article(full_text):
    """Structure-aware chunking: one chunk per numbered constitutional article."""
    article_pattern = re.compile(r'\n\s*(\d{1,3})\.\s+([A-Z][^:]{2,80}):')
    matches = list(article_pattern.finditer(full_text))

    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        article_text = full_text[start:end].strip()
        if len(article_text) > 20:
            chunks.append(article_text)
    return chunks