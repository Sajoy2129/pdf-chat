"""Load a PDF and turn it into plain text that Claude can read."""

import sys
from pathlib import Path

from pypdf import PdfReader


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Return the text of each page, in order (index 0 is page 1).

    Pages with no extractable text, such as scanned images, become an
    empty string, so page numbers stay correct.
    """
    reader = PdfReader(pdf_path)
    return [(page.extract_text() or "").strip() for page in reader.pages]


def join_pages(pages: list[str]) -> str:
    """Join page texts with '--- Page N ---' markers so Claude can cite pages."""
    return "\n\n".join(
        f"--- Page {number} ---\n{text}"
        for number, text in enumerate(pages, start=1)
    )


def load_pdf_text(pdf_path: str | Path) -> str:
    """Read a PDF and return all its text, with page markers.

    Raises ValueError if no page has any text, because sending Claude an
    empty document would give useless answers instead of a clear error.
    """
    pages = extract_pages(pdf_path)
    if not any(pages):
        raise ValueError(
            f"No text found in {pdf_path}. It may be a scanned PDF "
            "(images only), which needs OCR first."
        )
    return join_pages(pages)


if __name__ == "__main__":
    # Quick manual check: python -m pdf_chat.pdf_loader "docs/my file.pdf"
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m pdf_chat.pdf_loader <path-to-pdf>")
    text = load_pdf_text(sys.argv[1])
    page_count = text.count("--- Page ")
    print(f"{page_count} pages, {len(text):,} characters "
          f"(~{len(text) // 4:,} tokens)\n")
    print(text[:500])
