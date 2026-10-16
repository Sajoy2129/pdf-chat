"""A small web page to upload a PDF and ask Claude questions about it.

Run:
    flask --app app run --debug --port 5001
then open http://127.0.0.1:5001 in your browser. (Port 5000 is usually taken
by macOS's AirPlay Receiver, hence 5001.)

It reuses the same code as the terminal scripts: "Whole document" mode works
like simple_ask.py and "Search (RAG)" mode works like rag.py.
"""

import io
import os
import secrets
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, session, url_for
from pypdf.errors import PdfReadError
from rank_bm25 import BM25Okapi

from extract import extract_pages
from rag import Chunk, build_context, build_index, chunk_pages, pages_in, retrieve
from simple_ask import (
    NOT_FOUND,
    ask,
    build_document_text,
    build_system_prompt,
    format_pages,
)

load_dotenv()  # reads ANTHROPIC_API_KEY (and optional FLASK_SECRET_KEY) from .env

MAX_UPLOAD_MB = 20
MAX_DOCUMENTS = 20  # uploads kept in memory; the oldest is dropped after this

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
# The secret key signs the session cookie so visitors can't tamper with it.
# Without FLASK_SECRET_KEY in .env, a random one is made at each start, which
# logs everyone out (forgets their PDF) whenever the server restarts.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)


@dataclass
class Entry:
    """One question and its answer, shown in the page's history."""
    question: str
    mode: str
    answer: str
    source: str        # e.g. "page 3", or "" when nothing was found
    note: str          # e.g. "1 of 4 chunks sent"
    input_tokens: int | None


@dataclass
class Document:
    """An uploaded PDF, prepared once for both answering modes."""
    name: str
    pages: list[str]
    full_prompt: str      # whole document, for "full" mode
    chunks: list[Chunk]   # for "rag" mode
    index: BM25Okapi
    history: list[Entry] = field(default_factory=list)


# Uploaded documents live in server memory, keyed by a random id that is
# stored in each visitor's session cookie. Restarting the server clears them.
DOCUMENTS: dict[str, Document] = {}

_client = None


def get_client() -> anthropic.Anthropic:
    """Create the Claude client the first time it's needed, then reuse it."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def make_document(name: str, pages: list[str]) -> Document:
    chunks = chunk_pages(pages)
    return Document(
        name=name,
        pages=pages,
        full_prompt=build_system_prompt(build_document_text(pages)),
        chunks=chunks,
        index=build_index(chunks),
    )


def store_document(document: Document) -> str:
    """Keep the document in memory and return its new id."""
    doc_id = secrets.token_urlsafe(16)
    DOCUMENTS[doc_id] = document
    while len(DOCUMENTS) > MAX_DOCUMENTS:
        DOCUMENTS.pop(next(iter(DOCUMENTS)))  # dicts keep insertion order
    return doc_id


def current_document() -> Document | None:
    return DOCUMENTS.get(session.get("doc_id"))


def answer_question(document: Document, question: str, mode: str) -> Entry:
    """Answer with the whole document ("full") or the top BM25 chunks ("rag")."""
    if mode == "rag":
        results = retrieve(document.index, document.chunks, question)
        if not results:
            return Entry(question, mode, NOT_FOUND, "",
                         "no chunk matched the question's keywords; "
                         "no API call made", None)
        system_prompt = build_system_prompt(build_context(results))
        allowed_pages = pages_in(results)
        note = f"{len(results)} of {len(document.chunks)} chunks sent"
    else:
        system_prompt = document.full_prompt
        allowed_pages = set(range(1, len(document.pages) + 1))
        note = "whole document sent"

    reply = ask(get_client(), system_prompt, question, allowed_pages)
    source = format_pages(reply.pages) if reply.pages else ""
    return Entry(question, mode, reply.text, source, note, reply.input_tokens)


@app.get("/")
def index():
    return render_template("index.html", document=current_document(),
                           max_upload_mb=MAX_UPLOAD_MB)


@app.post("/upload")
def upload():
    file = request.files.get("pdf")
    if file is None or file.filename == "":
        flash("Please choose a PDF file first.")
        return redirect(url_for("index"))
    if not file.filename.lower().endswith(".pdf"):
        flash("That file isn't a PDF. Please choose a file ending in .pdf.")
        return redirect(url_for("index"))

    try:
        # Read the upload in memory; nothing is saved to disk.
        pages = extract_pages(io.BytesIO(file.read()))
    except PdfReadError:
        flash("Couldn't read that PDF. It may be damaged or password-protected.")
        return redirect(url_for("index"))
    if not any(pages):
        flash("No text found in that PDF. It may be scanned images, "
              "which need OCR first.")
        return redirect(url_for("index"))

    session["doc_id"] = store_document(make_document(file.filename, pages))
    return redirect(url_for("index"))


@app.post("/ask")
def ask_question():
    document = current_document()
    if document is None:
        flash("Please upload a PDF first.")
        return redirect(url_for("index"))

    question = request.form.get("question", "").strip()
    if not question:
        flash("Please type a question.")
        return redirect(url_for("index"))
    mode = "rag" if request.form.get("mode") == "rag" else "full"

    try:
        entry = answer_question(document, question, mode)
    except anthropic.AuthenticationError:
        flash("Invalid API key. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.RateLimitError:
        flash("Rate limited. Wait a moment and try again.")
    except anthropic.APIConnectionError:
        flash("Network error. Check your internet connection.")
    except anthropic.APIStatusError as error:
        flash(f"API error {error.status_code}: {error.message}")
    else:
        document.history.insert(0, entry)  # newest first

    # Redirect after POST, so refreshing the page doesn't ask (and pay) again.
    return redirect(url_for("index"))


@app.errorhandler(413)
def file_too_large(error):
    flash(f"That file is too large. The limit is {MAX_UPLOAD_MB} MB.")
    return redirect(url_for("index"))
