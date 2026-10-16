"""Ask Claude about a PDF using RAG: send only the most relevant chunks.

This is Version 2 from PLAN.md. Steps:
  1. Split the PDF's text into overlapping chunks of ~300 words.
  2. Build a BM25 keyword index over the chunks.
  3. For each question, find the top 5 chunks and send only those to Claude.

Usage:
    python rag.py                          # uses the only PDF in docs/
    python rag.py "docs/my file.pdf"
    python rag.py --show-chunks            # also print which chunks were picked
To quit, type quit / exit / q, press Enter on an empty line, or Ctrl+C / Ctrl+D.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from extract import extract_pages, find_pdf
from simple_ask import NOT_FOUND, ask_and_print, build_system_prompt, read_question

CHUNK_SIZE = 300   # words per chunk
OVERLAP = 50       # words shared between neighbouring chunks
TOP_K = 5          # how many chunks to send to Claude

# Very common words that say nothing about the topic. Dropping them stops
# "what is the total" from matching every chunk that contains "the".
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "of", "on", "or",
    "that", "the", "this", "to", "was", "what", "when", "where", "which",
    "who", "why", "will", "with", "you",
}


@dataclass
class Chunk:
    text: str             # plain words, used for keyword search
    text_with_pages: str  # same words with [Page N] markers, sent to Claude
    first_page: int
    last_page: int

    def pages(self) -> set[int]:
        """Every page this chunk has words from."""
        return set(range(self.first_page, self.last_page + 1))

    def label(self) -> str:
        if self.first_page == self.last_page:
            return f"page {self.first_page}"
        return f"pages {self.first_page}-{self.last_page}"


def add_page_markers(window: list[tuple[str, int]]) -> str:
    """Join words, inserting '[Page N]' wherever a new page starts.

    This lets Claude see which page each sentence is on, even when a chunk
    crosses a page break.
    """
    parts = []
    current_page = None
    for word, page in window:
        if page != current_page:
            parts.append(f"[Page {page}]")
            current_page = page
        parts.append(word)
    return " ".join(parts)


def chunk_pages(pages: list[str], size: int = CHUNK_SIZE,
                overlap: int = OVERLAP) -> list[Chunk]:
    """Split page texts into chunks of `size` words that overlap by `overlap`.

    Chunks may cross page boundaries; each chunk remembers which pages its
    words came from so Claude can still cite page numbers.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")

    # One (word, page number) pair per word in the whole document.
    words = [
        (word, page_number)
        for page_number, text in enumerate(pages, start=1)
        for word in text.split()
    ]

    step = size - overlap  # how far the window moves each time
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        chunks.append(Chunk(
            text=" ".join(word for word, _ in window),
            text_with_pages=add_page_markers(window),
            first_page=window[0][1],
            last_page=window[-1][1],
        ))
        if start + size >= len(words):  # this window reached the end
            break
    return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase, keep only letters/digits, and drop stopwords.

    Used for both chunks and questions, so they are compared the same way.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in STOPWORDS]


def build_index(chunks: list[Chunk]) -> BM25Okapi:
    """Build a BM25 keyword index with one entry per chunk."""
    return BM25Okapi([tokenize(chunk.text) for chunk in chunks])


def retrieve(index: BM25Okapi, chunks: list[Chunk], question: str,
             k: int = TOP_K) -> list[tuple[Chunk, float]]:
    """Return up to k (chunk, score) pairs, best first.

    Only chunks that share at least one word with the question are returned,
    so unrelated chunks are never sent as filler. We check shared words
    directly instead of using "score > 0": BM25Okapi gives a word that appears
    in half or more of the chunks a zero or negative weight, so a chunk that
    matches can still score <= 0 (always true when there is only one chunk).
    """
    question_words = tokenize(question)
    scores = index.get_scores(question_words)
    matching = [
        (chunk, score) for chunk, score in zip(chunks, scores)
        if set(question_words) & set(tokenize(chunk.text))
    ]
    matching.sort(key=lambda pair: pair[1], reverse=True)
    return matching[:k]


def build_context(results: list[tuple[Chunk, float]]) -> str:
    """Join the retrieved chunks; page markers inside each show its pages."""
    return "\n\n".join(
        f"--- Excerpt {number} ---\n{chunk.text_with_pages}"
        for number, (chunk, _) in enumerate(results, start=1)
    )


def pages_in(results: list[tuple[Chunk, float]]) -> set[int]:
    """All page numbers Claude was shown, so it can't cite any other page."""
    return set().union(*(chunk.pages() for chunk, _ in results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", nargs="?", help="PDF path (default: docs/)")
    parser.add_argument("--show-chunks", action="store_true",
                        help="print the chunks BM25 picked for each question")
    args = parser.parse_args()

    load_dotenv()  # reads ANTHROPIC_API_KEY from .env
    pdf_path = Path(args.pdf) if args.pdf else find_pdf()
    chunks = chunk_pages(extract_pages(pdf_path))
    index = build_index(chunks)
    client = anthropic.Anthropic()

    print(f"Loaded {pdf_path}: {len(chunks)} chunks of ~{CHUNK_SIZE} words "
          f"({OVERLAP}-word overlap). Type 'quit' to stop.")
    while True:
        question = read_question()
        if question is None:
            print("Goodbye!")
            break

        results = retrieve(index, chunks, question)
        if not results:
            # Nothing to answer from, so don't call Claude (saves money too).
            print(f"\n{NOT_FOUND}")
            print("\n[No chunk shared a keyword with the question; "
                  "no API call made]")
            continue

        if args.show_chunks:
            for chunk, score in results:
                print(f"  [{score:5.2f}] {chunk.label()}: {chunk.text[:80]}...")

        system_prompt = build_system_prompt(build_context(results))
        ask_and_print(client, system_prompt, question, pages_in(results),
                      extra_info=f" | chunks sent: {len(results)} of "
                                 f"{len(chunks)}")


if __name__ == "__main__":
    main()
