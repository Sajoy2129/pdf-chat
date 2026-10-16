"""Tests for the Flask web page (app.py).

Flask's test client sends fake browser requests, so no server is started.
Claude is replaced by a fake ask(), so the tests are free and offline.
"""

import io

import pytest

import app as web
from pdf_factory import build_pdf_bytes
from simple_ask import NOT_FOUND, Reply


@pytest.fixture
def client(monkeypatch):
    """A test browser, plus a fake Claude that records each call."""
    web.DOCUMENTS.clear()
    calls = []

    def fake_ask(client, system_prompt, question, allowed_pages):
        calls.append({"prompt": system_prompt, "question": question,
                      "allowed_pages": allowed_pages})
        return Reply("Net 30 days.", [2], 321)

    monkeypatch.setattr(web, "ask", fake_ask)
    monkeypatch.setattr(web, "get_client", lambda: None)
    web.app.config["TESTING"] = True
    with web.app.test_client() as test_client:
        test_client.calls = calls
        yield test_client


def upload(client, pages=("Quotation for lifebuoys", "Payment terms Net 30 days"),
           filename="quote.pdf", data=None):
    if data is None:
        data = build_pdf_bytes(list(pages))
    return client.post(
        "/upload",
        data={"pdf": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
        follow_redirects=True,
    )


def ask(client, question, mode="full"):
    return client.post("/ask", data={"question": question, "mode": mode},
                       follow_redirects=True)


# --- uploading ---------------------------------------------------------------

def test_home_page_shows_upload_form_but_no_question_form(client):
    html = client.get("/").get_data(as_text=True)

    assert "Upload a PDF" in html
    assert "Ask a question" not in html


def test_upload_pdf_shows_name_and_page_count(client):
    html = upload(client).get_data(as_text=True)

    assert "quote.pdf" in html
    assert "2 pages" in html
    assert "Ask a question" in html


def test_upload_without_file_shows_error(client):
    response = client.post("/upload", data={}, follow_redirects=True)

    assert "Please choose a PDF file first." in response.get_data(as_text=True)


def test_upload_non_pdf_is_rejected(client):
    html = upload(client, filename="notes.txt", data=b"hello").get_data(as_text=True)

    assert "isn&#39;t a PDF" in html  # Jinja escapes the apostrophe


def test_upload_broken_pdf_shows_error(client):
    html = upload(client, data=b"this is not really a pdf").get_data(as_text=True)

    assert "Couldn" in html and "read that PDF" in html


def test_upload_pdf_without_text_shows_ocr_hint(client):
    html = upload(client, pages=["", ""]).get_data(as_text=True)

    assert "No text found" in html


# --- asking ------------------------------------------------------------------

def test_ask_before_upload_shows_error(client):
    html = ask(client, "Anything?").get_data(as_text=True)

    assert "Please upload a PDF first." in html


def test_full_mode_sends_whole_document_and_shows_source(client):
    upload(client)

    html = ask(client, "What are the payment terms?").get_data(as_text=True)

    call = client.calls[0]
    assert "Quotation for lifebuoys" in call["prompt"]   # page 1 sent
    assert "Payment terms Net 30 days" in call["prompt"]  # page 2 sent
    assert call["allowed_pages"] == {1, 2}
    assert "Net 30 days." in html
    assert "Source: page 2" in html
    assert "321 input tokens" in html


def test_rag_mode_with_no_matching_keywords_skips_claude(client):
    upload(client)

    html = ask(client, "Who is the customer?", mode="rag").get_data(as_text=True)

    assert client.calls == []          # no API call made
    assert NOT_FOUND.replace("'", "&#39;") in html


def test_rag_mode_sends_matching_chunks(client):
    upload(client)

    ask(client, "What are the payment terms?", mode="rag")

    assert "Payment terms" in client.calls[0]["prompt"]


def test_question_html_is_escaped(client):
    upload(client)

    html = ask(client, "<script>alert(1)</script>").get_data(as_text=True)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_newest_answer_is_shown_first(client):
    upload(client)
    ask(client, "first question")

    html = ask(client, "second question").get_data(as_text=True)

    assert html.index("second question") < html.index("first question")
