"""Tests for rag.py. No API calls: chunking and BM25 run locally."""

import pytest

from rag import build_context, build_index, chunk_pages, retrieve, tokenize


def numbered_words(count: int, prefix: str = "w") -> str:
    """'w0 w1 w2 ...' so we can see exactly which words land in each chunk."""
    return " ".join(f"{prefix}{i}" for i in range(count))


def test_chunks_have_requested_size_and_overlap():
    chunks = chunk_pages([numbered_words(25)], size=10, overlap=3)

    words = [chunk.text.split() for chunk in chunks]
    assert words[0] == [f"w{i}" for i in range(0, 10)]
    assert words[1][0] == "w7"            # starts 3 words before w10
    assert words[0][-3:] == words[1][:3]  # last 3 words repeated


def test_chunks_cover_every_word_without_tiny_tail():
    chunks = chunk_pages([numbered_words(25)], size=10, overlap=3)

    # Starts at 0, 7, 14, 21 would leave a 4-word tail chunk at 21;
    # the window at 14 already reaches w23, so 21 is still needed for w24.
    assert chunks[-1].text.split()[-1] == "w24"
    assert len(chunks) == 4


def test_short_document_becomes_one_chunk():
    chunks = chunk_pages(["just a few words"], size=300, overlap=50)

    assert len(chunks) == 1
    assert chunks[0].text == "just a few words"


def test_chunks_remember_their_pages():
    pages = [numbered_words(6, "a"), numbered_words(6, "b")]

    chunks = chunk_pages(pages, size=8, overlap=2)

    assert (chunks[0].first_page, chunks[0].last_page) == (1, 2)
    assert chunks[0].label() == "pages 1-2"
    assert (chunks[1].first_page, chunks[1].last_page) == (2, 2)
    assert chunks[1].label() == "page 2"


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages(["some text"], size=10, overlap=10)


def test_tokenize_lowercases_strips_punctuation_and_stopwords():
    assert tokenize("What is the TOTAL price, in AED?") == [
        "total", "price", "aed"
    ]


def test_retrieve_finds_chunk_with_distinctive_word():
    pages = [
        "lifejackets and lifebuoys for the crew",
        "fire extinguisher and fire blanket",
        "payment terms net thirty days",
        "delivery to the shipyard",
    ]
    chunks = chunk_pages(pages, size=5, overlap=0)
    index = build_index(chunks)

    results = retrieve(index, chunks, "What are the payment terms?", k=2)

    # Chunks are cut by word count, so the best one may start on page 2;
    # what matters is that it holds the payment text from page 3.
    best = results[0][0]
    assert "payment terms" in best.text
    assert best.last_page == 3


def test_retrieve_skips_chunks_with_no_matching_words():
    chunks = chunk_pages(["alpha beta", "gamma delta", "epsilon zeta"],
                         size=2, overlap=0)
    index = build_index(chunks)

    assert retrieve(index, chunks, "completely unrelated question") == []


def test_build_context_labels_each_chunk():
    chunks = chunk_pages(["one two", "three four"], size=2, overlap=0)

    context = build_context([(chunks[1], 1.0), (chunks[0], 0.5)])

    assert context.startswith("--- Excerpt from page 2 ---\nthree four")
    assert "--- Excerpt from page 1 ---\none two" in context
