"""Ask Claude questions about a PDF by putting the whole document in the prompt.

This is Version 1 from PLAN.md (full context). Usage:
    python simple_ask.py                      # uses the only PDF in docs/
    python simple_ask.py "docs/my file.pdf"   # or name a PDF yourself
Type a question and press Enter. To quit, type quit / exit / q, press Enter
on an empty line, or press Ctrl+C or Ctrl+D.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from extract import extract_pages, find_pdf

MODEL = "claude-opus-5-5"

NOT_FOUND = "I couldn't find that in the document."

INSTRUCTIONS = f"""You answer questions using ONLY the document text below.

Rules:
- Use only facts written in the text. Do not use outside knowledge, and do
  not guess, assume, or fill gaps.
- If the text does not contain the answer, set "found" to false and
  "answer" to "{NOT_FOUND}"
- If it does, set "found" to true, answer briefly, and list in "pages" every
  page number your answer comes from. Pages are marked in the text like
  "--- Page 3 ---" or "[Page 3]"."""

# Structured output: Claude must reply with JSON in exactly this shape, so the
# code (not Claude's wording) decides what to print.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "pages": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["found", "answer", "pages"],
    "additionalProperties": False,
}

QUIT_WORDS = {"quit", "exit", "q"}


@dataclass
class Reply:
    text: str
    pages: list[int]
    input_tokens: int


def is_quit(text: str) -> bool:
    """True if the user typed nothing or a quit word (any capitalisation)."""
    return text.strip().lower() in QUIT_WORDS | {""}


def read_question() -> str | None:
    """Ask for a question. Return None when the user wants to quit.

    Ctrl+D raises EOFError and Ctrl+C raises KeyboardInterrupt; both mean
    "quit", so we catch them instead of showing an error trace.
    """
    try:
        question = input("\nQuestion: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()  # move past the ^C / ^D so "Goodbye!" starts on a new line
        return None
    return None if is_quit(question) else question


def build_document_text(pages: list[str]) -> str:
    """Join pages with '--- Page N ---' markers so Claude can cite pages."""
    return "\n\n".join(
        f"--- Page {number} ---\n{text}"
        for number, text in enumerate(pages, start=1)
    )


def build_system_prompt(document_text: str) -> str:
    """Put the instructions first, then the document text."""
    return f"{INSTRUCTIONS}\n\n<document>\n{document_text}\n</document>"


def parse_reply(raw_json: str, allowed_pages: set[int],
                input_tokens: int) -> Reply:
    """Turn Claude's JSON into a Reply, enforcing the "no guessing" rules.

    - found is false            -> the fixed NOT_FOUND message
    - cited pages not provided  -> dropped (Claude can't cite text it never saw)
    - no valid page left        -> NOT_FOUND, because the answer can't be
                                   traced back to the document
    """
    data = json.loads(raw_json)
    pages = sorted({page for page in data["pages"] if page in allowed_pages})
    if not data["found"] or not data["answer"].strip() or not pages:
        return Reply(NOT_FOUND, [], input_tokens)
    return Reply(data["answer"].strip(), pages, input_tokens)


def ask(client, system_prompt: str, question: str,
        allowed_pages: set[int]) -> Reply:
    """Send one question to Claude and return its checked Reply.

    allowed_pages is the set of page numbers Claude was shown; any other
    page it cites is ignored.
    """
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
        },
        # If a safety check declines the request, retry it on another model
        # instead of failing.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    input_tokens = response.usage.input_tokens

    if response.stop_reason == "refusal":
        return Reply("(Claude declined to answer this question.)", [],
                     input_tokens)

    # The reply is a list of blocks; the JSON is in the text block(s).
    raw_json = "".join(
        block.text for block in response.content if block.type == "text"
    )
    try:
        return parse_reply(raw_json, allowed_pages, input_tokens)
    except (json.JSONDecodeError, KeyError):
        # Happens if the answer was cut off (stop_reason "max_tokens").
        return Reply("(The answer was incomplete. Please try again.)", [],
                     input_tokens)


def format_pages(pages: list[int]) -> str:
    """[3] -> 'page 3';  [2, 3] -> 'pages 2, 3'."""
    word = "page" if len(pages) == 1 else "pages"
    return f"{word} {', '.join(str(page) for page in pages)}"


def format_reply(reply: Reply) -> str:
    """The answer, then where it came from (if anywhere)."""
    if not reply.pages:
        return reply.text
    return f"{reply.text}\n\nSource: {format_pages(reply.pages)}"


def ask_and_print(client, system_prompt: str, question: str,
                  allowed_pages: set[int], extra_info: str = "") -> None:
    """Ask Claude, handle API errors, and print the answer and token count."""
    try:
        reply = ask(client, system_prompt, question, allowed_pages)
    except anthropic.AuthenticationError:
        sys.exit("Invalid API key. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.RateLimitError:
        print("Rate limited. Wait a moment and try again.")
        return
    except anthropic.APIConnectionError:
        print("Network error. Check your internet connection.")
        return
    except anthropic.APIStatusError as error:
        print(f"API error {error.status_code}: {error.message}")
        return

    print(f"\n{format_reply(reply)}")
    print(f"\n[Input tokens: {reply.input_tokens:,}{extra_info}]")


def main() -> None:
    load_dotenv()  # reads ANTHROPIC_API_KEY from .env
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_pdf()
    pages = extract_pages(pdf_path)
    system_prompt = build_system_prompt(build_document_text(pages))
    all_pages = set(range(1, len(pages) + 1))
    client = anthropic.Anthropic()

    print(f"Loaded {pdf_path}. Ask a question (type 'quit' to stop).")
    while True:
        question = read_question()
        if question is None:
            print("Goodbye!")
            break
        ask_and_print(client, system_prompt, question, all_pages)


if __name__ == "__main__":
    main()
