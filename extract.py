"""Pull the text out of a PDF page by page and report page and word counts.

Usage:
    python extract.py                      # uses the only PDF in docs/
    python extract.py "docs/my file.pdf"   # or name a PDF yourself
"""

import sys
from pathlib import Path

from pypdf import PdfReader


def find_pdf(folder: str | Path = "docs") -> Path:
    """Return the single PDF in `folder`, or raise a clear error."""
    pdfs = sorted(Path(folder).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {folder}/")
    if len(pdfs) > 1:
        names = ", ".join(pdf.name for pdf in pdfs)
        raise ValueError(
            f"Several PDFs in {folder}/ ({names}). Pass one as an argument."
        )
    return pdfs[0]


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Return the text of each page, in order (index 0 is page 1).

    Pages with no extractable text, such as scanned images, become an
    empty string, so page numbers stay correct.
    """
    reader = PdfReader(pdf_path)
    return [(page.extract_text() or "").strip() for page in reader.pages]


def count_words(text: str) -> int:
    """Count words by splitting on any whitespace (spaces, tabs, newlines)."""
    return len(text.split())


def main() -> None:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_pdf()
    pages = extract_pages(pdf_path)

    print(f"File: {pdf_path}\n")
    for number, text in enumerate(pages, start=1):
        print(f"Page {number}: {count_words(text):,} words")

    total_words = sum(count_words(text) for text in pages)
    print(f"\nPages: {len(pages)}")
    print(f"Words: {total_words:,}")

    empty_pages = [n for n, text in enumerate(pages, start=1) if not text]
    if empty_pages:
        print(f"\nWarning: no text on pages {empty_pages}. "
              "They may be scanned images that need OCR.")


if __name__ == "__main__":
    main()
