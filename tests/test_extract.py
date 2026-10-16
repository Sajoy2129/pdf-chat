"""Tests for extract.py."""

import pytest

from extract import count_words, find_pdf


def test_count_words_normal_sentence():
    assert count_words("The quick brown fox") == 4


def test_count_words_ignores_extra_whitespace():
    assert count_words("  one\n\ntwo\tthree  ") == 3


def test_count_words_empty_string_is_zero():
    assert count_words("") == 0


def test_find_pdf_returns_the_only_pdf(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    assert find_pdf(tmp_path) == pdf


def test_find_pdf_raises_when_folder_has_no_pdf(tmp_path):
    with pytest.raises(FileNotFoundError, match="No PDF found"):
        find_pdf(tmp_path)


def test_find_pdf_raises_when_several_pdfs(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")

    with pytest.raises(ValueError, match="Several PDFs"):
        find_pdf(tmp_path)
