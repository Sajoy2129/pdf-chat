# PDF Chat

Ask questions about a PDF using the Claude API, either from the terminal or a simple web page.

Answers come **only from the document**. Every answer shows the page it came from, and when the
answer isn't in the document you get *"I couldn't find that in the document."* instead of a guess.

The project builds the same tool in two ways, so you can compare them:

| | **Whole document** (`simple_ask.py`) | **Search / RAG** (`rag.py`) |
|---|---|---|
| What is sent to Claude | The entire PDF text | Only the top 5 matching chunks (~300 words each) |
| Accuracy | Best: sees everything, understands reworded questions | Only finds chunks that share **exact words** with the question |
| Cost per question | Grows with document size | Stays small, whatever the document size |
| Best for | Short and medium documents | Large documents, or many of them |

For the full comparison (cost, speed, accuracy, size limits) and the build plan, see
[PLAN.md](PLAN.md).

## Setup

You need Python 3.10+ and an [Anthropic API key](https://console.anthropic.com/).

```bash
python3 -m venv .venv                 # create a virtual environment (once)
source .venv/bin/activate             # turn it on (every new terminal)
pip install -r requirements.txt       # install the dependencies
```

Create a file named `.env` in the project folder with your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is in `.gitignore`, so the key is never committed.

Put your PDF in the `docs/` folder (also gitignored). The terminal scripts use the only PDF there,
or you can pass a path.

## Usage

> Each question is a real API call and costs money (around 1–3 cents for a short PDF).

### Web page

```bash
flask --app app run --debug --port 5001
```

Open <http://127.0.0.1:5001>, upload a PDF, choose **Whole document** or **Search (RAG)**, and ask.
Use port 5001 because macOS's AirPlay Receiver usually takes port 5000.

### Terminal

```bash
python extract.py                      # how many pages and words the PDF has
python simple_ask.py                   # ask with the whole document
python rag.py                          # ask with search (RAG)
python rag.py --show-chunks            # also show which chunks the search picked
python simple_ask.py "docs/other.pdf"  # use a specific PDF
```

Type `quit`, `exit` or `q` (or press Enter on an empty line, Ctrl+C or Ctrl+D) to stop.

Example:

```
Question: What are the payment terms?

Net 30 days.

Source: page 4

[Input tokens: 962 | chunks sent: 1 of 4]
```

## How it works

1. **Extract:** `pypdf` reads the text of each page.
2. **Choose what to send:**
   - *Whole document:* every page, marked `--- Page N ---`.
   - *RAG:* the text is split into ~300-word chunks that overlap by 50 words, so a sentence cut
     at a chunk edge still appears whole in one chunk. The [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
     keyword search (`rank-bm25`) then picks the 5 best chunks for the question. If no chunk
     shares a word with the question, no API call is made.
3. **Ask Claude** (`claude-opus-5-5`). Claude must reply in a fixed JSON shape:
   `{"found": true/false, "answer": "...", "pages": [..]}`.
4. **Check before showing:** if `found` is false, the program shows the fixed "couldn't find"
   message. Pages Claude wasn't shown are dropped, and an answer with no valid page is treated
   as not found.

## Project structure

```
extract.py        PDF -> text per page; page and word counts
simple_ask.py     whole-document mode + shared code (Claude call, answer checks, quit handling)
rag.py            chunking, BM25 search, RAG mode
app.py            Flask web page (templates/index.html)
tests/            pytest tests (Claude is faked, so they're free and offline)
PLAN.md           approach comparison and step-by-step build plan
CLAUDE.md         notes for Claude Code when working on this repo
```

## Running tests

```bash
python -m pytest                       # all tests
python -m pytest tests/test_rag.py -v  # one file, with test names
python -m pytest tests/test_rag.py::test_retrieve_finds_the_only_chunk_of_a_short_document
```

Use `python -m pytest` rather than plain `pytest`, so Python can find the project's modules.

## Limitations

- **Keyword search misses synonyms.** In RAG mode, "Who is the customer?" finds nothing if the
  PDF only gives the company's name. Reword using the document's own words, or use whole-document
  mode. Embeddings (search by meaning) would fix this; see "Stretch ideas" in PLAN.md.
- **Scanned PDFs** (images of text) have no extractable text and need OCR first.
- **The web page is for local use.** Uploaded PDFs are kept in memory and are lost when the
  server restarts. It uses Flask's development server, which isn't meant for the public internet.
