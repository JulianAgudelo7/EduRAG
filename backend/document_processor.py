
from pypdf import PdfReader


def extract_text_from_pdf(file_path):
    """
    Extrae el texto de un documento PDF.
    Devuelve una lista con el contenido de cada página.
    """

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "content": text
            })

    return pages
