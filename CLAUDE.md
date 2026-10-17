# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning project that answers questions about a PDF with the Claude API in two ways, so they
can be compared (see `PLAN.md` for the roadmap and the cost/speed/accuracy comparison):

- `simple_ask.py`: **full context**, where the whole PDF text goes in the system prompt.
- `rag.py`: **RAG**, which uses ~300-word chunks with 50-word overlap, `rank-bm25` keyword search,
  and sends only the top 5 chunks.

## Commands

```bash
source .venv/bin/activate              # always use the project venv
pip install -r requirements.txt

python extract.py                      # page + word counts for the PDF in docs/
python simple_ask.py                   # v1 question loop (quit / exit / q to stop)
python rag.py --show-chunks            # v2 question loop, prints BM25's picks and scores
python simple_ask.py "docs/other.pdf"  # any script accepts an explicit PDF path
flask --app app run --debug --port 5001  # web UI at http://127.0.0.1:5001 (macOS AirPlay holds 5000)

python -m pytest                       # all tests
python -m pytest tests/test_rag.py::test_retrieve_finds_chunk_with_distinctive_word
```

- Use `python -m pytest`, not bare `pytest`. The modules live at the repo root and there is no
  `pytest.ini`/`conftest.py`, so `python -m` is what puts the root on `sys.path`.
- Tests never call the API. Running the scripts does, and **costs real money** (a few cents per
  question). Ask before running them, and pipe questions in for non-interactive runs:
  `printf 'What are the payment terms?\nquit\n' | python rag.py`.
- `.env` holds `ANTHROPIC_API_KEY`. `docs/` (the PDFs) is gitignored. Without an argument the
  scripts expect exactly one PDF in `docs/`.

## Architecture

Flat modules at the repo root, with a one-way import chain:

```
extract.py  ->  simple_ask.py  ->  rag.py  ->  app.py (Flask, templates/index.html)
```

- `app.py` holds no answering logic of its own. Its "full" and "rag" modes call the same `ask()`,
  `retrieve()`, `build_context()` and `pages_in()` as the scripts. Uploads are parsed in memory
  (never written to disk) and kept in the module-level `DOCUMENTS` dict, keyed by a random id in
  the signed session cookie. A server restart clears them. Set `FLASK_SECRET_KEY` in `.env` to
  keep sessions across restarts. Tests monkeypatch `app.ask` and `app.get_client`, and build PDFs
  with `tests/pdf_factory.py`, which pytest's default import mode puts on `sys.path`.

- `extract.py`: `find_pdf()`, `extract_pages()` (pypdf; one string per page, `""` for pages
  without text so page numbers stay aligned).
- `simple_ask.py` is **both** the v1 script and the shared library: the Claude call (`ask`),
  printing and error handling (`ask_and_print`), the quit logic (`read_question`), the system prompt,
  and `NOT_FOUND`. `rag.py` imports these. Change shared behaviour there, not in `rag.py`.

### The answer contract (spans both scripts)

Answers must come only from the provided text and must cite pages. This is enforced in code, not
just in the prompt:

1. `ask()` sends `output_config.format` with `ANSWER_SCHEMA`, so Claude must return JSON
   `{"found": bool, "answer": str, "pages": [int]}`.
2. `parse_reply()` returns the fixed `NOT_FOUND` string ("I couldn't find that in the document.")
   when `found` is false. It drops cited pages that are not in `allowed_pages`, and treats an
   answer with no valid page as not found.
3. Callers pass `allowed_pages`: all pages in `simple_ask.py`, or `pages_in(results)` (the pages of
   the chunks actually sent) in `rag.py`.
4. `format_reply()` adds `Source: page N`.

In `rag.py`, each `Chunk` has two texts: `text` (plain, used for BM25) and `text_with_pages`
(`[Page N]` markers at page breaks, sent to Claude so it can cite the exact page when a chunk
crosses pages). `retrieve()` decides "match" by shared tokens and uses the BM25 score only for
ranking. Never filter on `score > 0`: `BM25Okapi` gives words that appear in ≥ half of the chunks
a weight ≤ 0, so a single-chunk document would never match. With no matching chunk, callers
return `NOT_FOUND` without calling the API.

### Claude API usage

- Model `claude-opus-5-5`, called via `client.beta.messages.create` with
  `betas=["server-side-fallback-2026-07-01"]` and `fallbacks="default"` (retries on another model
  if a safety check declines). Effort is set explicitly to `medium`.
- On this model, thinking can't be disabled and `budget_tokens` / forced `tool_choice` return a 400.
  The response contains thinking blocks, so read only `type == "text"` blocks.
- `ask()` handles `stop_reason == "refusal"` and truncated JSON (`max_tokens`).
- Tests replace the client with a `SimpleNamespace` fake exposing `beta.messages.create`. If the
  call path changes (e.g. to non-beta `messages.create` or `.parse()`), update
  `make_fake_client()` in `tests/test_simple_ask.py`.

## Git

Remote `origin` is the private GitHub repo `Sajoy2129/pdf-chat`. Stage files explicitly
(`git add <files>`), not `git add -A`. Never commit `.env` or anything in `docs/` (real customer
PDFs), and keep real document details (names, reference numbers) out of code, tests and docs.
