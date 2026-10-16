# PDF Chat: Project Plan

Ask questions about a PDF using the Claude API. We build it twice to learn the trade-offs:

- **Version 1, full context:** put the whole document's text in the prompt.
- **Version 2, RAG with BM25:** split the document into chunks, find the best chunks with keyword
  search, and send only those.

## Key terms

- **Token:** the unit an LLM reads and is billed by, about ¾ of an English word (~4 characters).
- **Context window:** the most tokens Claude can read in one request (1M for `claude-opus-5-5`).
- **Prompt caching:** Anthropic keeps the unchanged *start* of your prompt for 5 minutes (or 1 hour
  if you ask). Repeat requests read it from the cache at 5% of the normal input price. The first
  request pays 1.25× to write it.
- **Thinking tokens:** Opus 5.5 always thinks a little before answering. These tokens are billed
  as output. You control how much it thinks with `effort` (`low` … `max`, default `medium`).
- **RAG (Retrieval-Augmented Generation):** first *retrieve* the relevant pieces, then let the model
  *generate* an answer from only those pieces.
- **Chunk:** a small piece of the document (e.g. ~300 words) that we search over.
- **BM25:** a classic keyword-ranking formula used by search engines. A chunk scores higher when it
  contains the question's words, especially rare ones. No AI involved; it runs locally and is free.

## Comparison

Prices for `claude-opus-5-5`: input $4 / 1M tokens, output $20 / 1M, cache write $5 / 1M,
cache read $0.20 / 1M. Our sample PDF (measured with `extract.py`) is **4 pages, 930 words,
≈1,250 tokens**; step 4 gets the exact count with `count_tokens`. Output (answer + thinking) is
about the same for both versions, roughly 500–1,500 tokens ≈ $0.01–0.03 per question.

| | **v1: Whole text in prompt** | **v2: BM25 RAG** |
|---|---|---|
| **Cost (input) per question** | Grows in proportion to document size. Our PDF: ≈ $0.005 (1,250 × $4 / 1M). A 100-page PDF (~30K tokens): ≈ $0.12 per question, or ≈ $0.006 for follow-ups once cached. Our PDF may be too short to cache at all (the minimum is 512–4,096 tokens depending on the model), so expect `cache_read_input_tokens == 0`, which is fine at this size. | ~5 chunks ≈ 2K tokens ≈ $0.008, whatever the document size. That's *more* than v1 for our small PDF: RAG only saves money once the document is much bigger than the chunks you send. |
| **Speed** | Claude must read every token, so time-to-first-word grows with the document. Caching cuts this a lot on follow-ups. | BM25 search takes milliseconds and the prompt is small, so it stays fast even for huge documents. |
| **Accuracy** | Best. Claude sees everything, so summaries, "compare section 2 with 5" and reworded questions all work. Very long contexts can make details slightly easier to miss. | Only as good as the retrieval. BM25 matches exact words, so it misses synonyms ("revenue" vs "income"). It can't summarize the whole document, and a chunk boundary can split an answer. |
| **Document-size limit** | Hard limit: 1M tokens of context, minus room for the answer. That's roughly 1,500–2,500 pages of plain text. Cost and wait time hurt well before that. | Effectively none. Only the top-k chunks are sent, so it scales to huge documents or many PDFs. Only indexing time and memory grow. |
| **Complexity** | ~40 lines. Nothing to tune. | Chunking, tokenizing, indexing, and tuning (chunk size, overlap, k). |

**Rule of thumb:** if the document fits comfortably, start with full context. It's simpler and
more accurate. Switch to RAG when documents get large or numerous, or when per-question cost
matters at scale.

**For our 4-page PDF, v1 is clearly the right choice.** We still build v2 to learn how RAG works.
To *see* the difference, also try both versions on a larger PDF (100+ pages).

## Setup

```bash
source .venv/bin/activate          # turn on the project's virtual environment
pip install -r requirements.txt    # anthropic, pypdf, rank-bm25, python-dotenv, pytest
```

- Put your key in `.env` as `ANTHROPIC_API_KEY=...` (gitignored, so never committed).
- Put PDFs in `docs/` (also gitignored).

## Project layout

```
pdf_chat/
  pdf_loader.py     # extract_pages(), join_pages(), load_pdf_text()
  claude_client.py  # ask_claude(context, question) -> answer + usage (shared by v1 and v2)
  full_context.py   # v1: answer(pdf_path, question)
  rag.py            # v2: chunk_pages(), tokenize(), build_index(), retrieve(), answer()
extract.py          # DONE: page/word counts for a PDF
main.py             # CLI: python main.py "docs/file.pdf" --mode full|rag
tests/              # pytest; the Claude call is mocked so tests are free and fast
```

## Version 1: Full context (one small commit per step)

1. **Load the PDF:** `pdf_loader.py` extracts text page by page with `pypdf` and adds
   `--- Page N ---` markers so Claude can cite pages. Raises a clear error on scanned (image-only)
   PDFs. *(Already written in commit `e5fd8aa`. `git restore pdf_chat tests pytest.ini` brings it back.)*
2. **Call Claude:** `claude_client.py`
   - `load_dotenv()`, then `anthropic.Anthropic()` reads the key from the environment.
   - `client.messages.create(model="claude-opus-5-5", max_tokens=16000,
     output_config={"effort": "medium"}, ...)`. Set effort explicitly so it's easy to experiment.
   - Document goes in `system` as a text block with `cache_control: {"type": "ephemeral"}`.
     Keep it byte-identical between questions or the cache misses.
   - Instructions: answer only from the document, say "not in the document" if missing, cite pages.
   - Check `response.stop_reason` (e.g. `"refusal"`, `"max_tokens"`) before reading the text.
   - Print `response.usage` (`input_tokens`, `cache_creation_input_tokens`,
     `cache_read_input_tokens`, `output_tokens`) and the time taken, so you can *see* cost and caching.
   - Handle errors with specific SDK exceptions (`AuthenticationError`, `RateLimitError`,
     `APIConnectionError`).
   - Test: mock the client and check the request contains the document, question and cache_control.
3. **CLI:** `full_context.py` + `main.py --mode full`. Load the PDF once, then loop asking questions.
4. **Size guard:** call `client.messages.count_tokens(...)` once and print the real token count and
   estimated cost. Stop with a clear message if it's over ~900K tokens. Never silently cut text.

## Version 2: RAG with BM25

5. **Chunking:** `chunk_pages(pages, size=300, overlap=50)` (in words). Overlap repeats the end of
   one chunk at the start of the next, so answers are less likely to be split. Each chunk keeps
   its page number. Tests: sizes, overlap, page numbers, empty pages skipped.
6. **Index:** `tokenize(text)` lowercases, strips punctuation and drops stopwords ("the", "is").
   Use the *same* function for chunks and questions. `build_index()` returns
   `rank_bm25.BM25Okapi(tokenized_chunks)`. Test: the index builds and scores one number per chunk.
7. **Retrieve:** `retrieve(question, k=5)` returns the top-5 chunks with their scores. If every score
   is 0 (no shared words), say so. Don't send random chunks. Test: a distinctive word from
   page N retrieves a chunk from page N.
8. **Answer:** `rag.answer()` sends only the retrieved chunks (labelled with page numbers) to the
   same `ask_claude()`. Add `main.py --mode rag` and `--show-chunks` to print what BM25 picked,
   which is the first thing to check when an answer is wrong.

## Compare the two versions

Write 6–8 questions about the PDF:

- 2–3 using the document's exact wording (BM25 should do well),
- 2–3 reworded with different words (BM25 may struggle),
- 1 "summarize the whole document" question (RAG should struggle),
- 1 question the document can't answer (both should say so).

Run each question in both modes and record the results in `RESULTS.md`:

| Question | Mode | Correct? | Input tokens | Output tokens | Cost | Seconds |
|---|---|---|---|---|---|---|

## Stretch ideas

- **Hybrid search:** combine BM25 with embeddings (meaning-based search) to catch synonyms.
- **Native PDF input:** send the PDF file itself as a `document` block with `citations` enabled.
  Claude then also sees tables and images and returns exact page citations (limit: 600 pages / 32 MB).
- **Multiple PDFs:** index a whole folder, which is where RAG really shines.

## How to run tests

```bash
source .venv/bin/activate
pytest -v
```
