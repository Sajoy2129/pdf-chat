# PDF Chat — Project Plan

Ask questions about a PDF using the Claude API. We build it twice to learn the trade-offs:

- **Version 1 — Full context:** put the whole document's text in the prompt.
- **Version 2 — RAG with BM25:** split the document into chunks, find the most relevant chunks
  with keyword search, and send only those.

## Key terms

- **Token** — the unit an LLM reads and is billed by. Roughly ¾ of an English word (~4 characters).
- **Context window** — the maximum number of tokens Claude can read in one request (1M for Opus 5.5).
- **Prompt caching** — Anthropic stores the unchanged *start* of your prompt for a few minutes.
  Repeat requests that start with the same text read it from the cache at ~5% of the normal price.
- **RAG (Retrieval-Augmented Generation)** — first *retrieve* relevant pieces of a document, then
  let the model *generate* an answer from only those pieces.
- **Chunk** — a small piece of the document (e.g. ~300 words) that we search over.
- **BM25** — a classic keyword-ranking formula (used by search engines for decades). It scores a
  chunk higher when it contains the question's words, especially rare words. No AI involved.

## Comparison

Prices for `claude-opus-5-5`: $4 / 1M input tokens, $20 / 1M output tokens, cache reads $0.20 / 1M,
cache writes ~1.25× input. Numbers below use our sample PDF: 31 pages, ~24K tokens.

| | **v1: Whole text in prompt** | **v2: BM25 RAG** |
|---|---|---|
| **Cost per question** | ~$0.10 input uncached. ~$0.005 with prompt caching after the first question. Grows linearly with document size. | ~2K tokens of chunks → ~$0.008. Barely depends on document size. |
| **Speed** | Slower to start answering — Claude reads all 24K tokens every time (caching helps). | Fast — BM25 search runs locally in milliseconds; the prompt is small. |
| **Accuracy** | Best. Claude sees everything: summaries, "compare section 2 with 5", paraphrased questions all work. | Only as good as retrieval. BM25 matches exact words, so it misses synonyms ("revenue" vs "income"), can't summarize the whole document, and a chunk boundary can split an answer. |
| **Document-size limit** | ~1M tokens (roughly 2,000+ text pages). Cost and latency become painful well before that. | Effectively unlimited — only the top-k chunks are sent. Scales to huge or many PDFs. |
| **Complexity** | ~40 lines of code. | Chunking, indexing and tuning (chunk size, overlap, k). |

**Rule of thumb:** if the document fits comfortably, start with full context — it's simpler and
more accurate. Move to RAG when documents get large or numerous, or per-question cost matters.

## Setup

```bash
source .venv/bin/activate          # turn on the project's virtual environment
pip install -r requirements.txt    # anthropic, pypdf, rank-bm25, python-dotenv, pytest
```

- Put your key in `.env` as `ANTHROPIC_API_KEY=...` (already gitignored — never commit it).
- Put PDFs in `docs/` (also gitignored).
- Add `pytest` to `requirements.txt` when we write the first test.

## Project layout

```
pdf_chat/
  pdf_loader.py     # extract_pages(path) -> list[str]   (pypdf)
  claude_client.py  # ask_claude(context, question) -> str  (shared by v1 and v2)
  full_context.py   # v1: answer(pdf_path, question)
  rag.py            # v2: chunk_text(), build_index(), retrieve(), answer()
main.py             # CLI: python main.py docs/file.pdf --mode full|rag
tests/              # pytest; the Claude call is mocked so tests are free and fast
```

## Version 1 — Full context

One small git commit per step.

1. **Load the PDF** — `pdf_loader.py`: extract text page by page with `pypdf`, joining pages with
   `--- Page N ---` markers so Claude can cite page numbers. Test: page count and non-empty text.
2. **Call Claude** — `claude_client.py`:
   - `anthropic.Anthropic()` reads the key from the environment (`load_dotenv()` first).
   - `client.messages.create(model="claude-opus-5-5", max_tokens=16000, ...)`.
   - Document goes in the `system` prompt with `cache_control: {"type": "ephemeral"}` so
     follow-up questions hit the cache.
   - Instructions: answer only from the document, say so if the answer isn't there, cite pages.
   - Print `response.usage` (`input_tokens`, `cache_read_input_tokens`, `output_tokens`) so you
     can *see* the cost and the cache working.
3. **CLI** — `full_context.py` + `main.py --mode full`: load once, then loop asking questions.
4. **Size guard** — call `client.messages.count_tokens(...)` before the first question and stop
   with a clear message if the document is too large. Never silently cut the text.

## Version 2 — RAG with BM25

5. **Chunking** — `chunk_text(pages, size=300, overlap=50)` (word counts). Overlap means the end
   of one chunk repeats at the start of the next, so an answer is less likely to be split.
   Each chunk remembers its page number. Test: chunk sizes, overlap, page numbers.
6. **Index** — tokenize each chunk (lowercase, strip punctuation, drop stopwords like "the"),
   then build `rank_bm25.BM25Okapi(tokenized_chunks)`. Test: index builds, scores are numbers.
7. **Retrieve** — `retrieve(question, k=5)` returns the 5 highest-scoring chunks.
   Test: a distinctive word from page N retrieves a chunk from page N.
8. **Answer** — `rag.answer()` sends only the retrieved chunks to the same `ask_claude()`.
   Add `main.py --mode rag` and a `--show-chunks` flag to print what BM25 picked (great for
   debugging bad answers).

## Compare the two versions

Write 5–8 questions about the PDF:

- 2–3 that use the document's exact wording (BM25 should do well),
- 2–3 paraphrased with different words (BM25 may struggle),
- 1 "summarize the whole document" question (RAG should struggle).

Run each in both modes and record the results in `RESULTS.md`:

| Question | Mode | Correct? | Input tokens | Seconds |
|---|---|---|---|---|

## Stretch ideas

- **Hybrid search:** combine BM25 with embeddings (meaning-based search) to catch synonyms.
- **Native PDF input:** send the PDF itself as a `document` block with `citations` enabled, so
  Claude also sees tables/images and returns exact page citations.
- **Multiple PDFs:** index a whole folder — where RAG really shines.

## How to run tests

```bash
source .venv/bin/activate
pytest -v
```
