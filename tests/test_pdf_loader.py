"""Tests for pdf_chat.pdf_loader."""

import pytest

from pdf_chat.pdf_loader import extract_pages, join_pages, load_pdf_text


def test_extract_pages_returns_one_string_per_page(make_pdf):
    pdf_path = make_pdf(["Hello from page one", "Second page text"])

    pages = extract_pages(pdf_path)

    assert len(pages) == 2
    assert pages[0] == "Hello from page one"
    assert pages[1] == "Second page text"


def test_blank_page_becomes_empty_string_and_keeps_numbering(make_pdf):
    pdf_path = make_pdf(["First", "", "Third"])

    pages = extract_pages(pdf_path)

    assert pages == ["First", "", "Third"]


def test_join_pages_adds_page_markers():
    text = join_pages(["alpha", "beta"])

    assert text == "--- Page 1 ---\nalpha\n\n--- Page 2 ---\nbeta"


def test_load_pdf_text_includes_markers_and_text(make_pdf):
    pdf_path = make_pdf(["Invoice total 42", "Thank you"])

    text = load_pdf_text(pdf_path)

    assert "--- Page 1 ---" in text
    assert "--- Page 2 ---" in text
    assert "Invoice total 42" in text


def test_load_pdf_text_rejects_pdf_without_text(make_pdf):
    pdf_path = make_pdf(["", ""])

    with pytest.raises(ValueError, match="No text found"):
        load_pdf_text(pdf_path)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pages(tmp_path / "does_not_exist.pdf")
