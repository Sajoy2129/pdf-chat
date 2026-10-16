"""Build tiny, valid text PDFs for tests (no real PDF or extra library needed).

Copied from the old tests/conftest.py in commit e5fd8aa.
"""


def build_pdf_bytes(page_texts: list[str]) -> bytes:
    """Build a minimal, valid PDF with one line of text per page.

    Writing the PDF by hand means the tests need no extra library and no
    real PDF in docs/. Keep texts to plain letters, digits and spaces;
    parentheses and backslashes have special meaning in PDF strings.
    """
    page_count = len(page_texts)
    # Object numbers: 1 = catalog, 2 = page list, 3 = font,
    # then each page uses two objects: the page and its content stream.
    page_ids = [4 + 2 * i for i in range(page_count)]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)

    objects = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_id, text in zip(page_ids, page_texts):
        content_id = page_id + 1
        objects[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
        )

    # The cross-reference table at the end records each object's byte
    # offset, so we track offsets while writing the objects.
    pdf = b"%PDF-1.4\n"
    offsets = []
    for number in range(1, len(objects) + 1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n{objects[number]}\nendobj\n".encode()

    xref_start = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode()
    return pdf
