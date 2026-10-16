"""Ask Claude questions about a PDF by putting the whole document in the prompt.

This is Version 1 from PLAN.md (full context). Usage:
    python simple_ask.py                      # uses the only PDF in docs/
    python simple_ask.py "docs/my file.pdf"   # or name a PDF yourself
Type a question and press Enter. An empty line (or Ctrl+D) quits.
"""

import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from extract import extract_pages, find_pdf

MODEL = "claude-opus-5-5"

INSTRUCTIONS = """You answer questions about the document below.
Use only the document. If the answer is not in it, say so.
Mention the page number(s) your answer comes from."""


def build_document_text(pages: list[str]) -> str:
    """Join pages with '--- Page N ---' markers so Claude can cite pages."""
    return "\n\n".join(
        f"--- Page {number} ---\n{text}"
        for number, text in enumerate(pages, start=1)
    )


def build_system_prompt(document_text: str) -> str:
    """Put the instructions first, then the whole document."""
    return f"{INSTRUCTIONS}\n\n<document>\n{document_text}\n</document>"


def ask(client, system_prompt: str, question: str) -> tuple[str, int]:
    """Send one question to Claude. Return (answer, input token count)."""
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        # If a safety check declines the request, retry it on another model
        # instead of failing.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )

    if response.stop_reason == "refusal":
        answer = "(Claude declined to answer this question.)"
    else:
        # The reply is a list of blocks; keep only the text ones.
        answer = "".join(
            block.text for block in response.content if block.type == "text"
        )
        if response.stop_reason == "max_tokens":
            answer += "\n(Answer was cut off: it hit max_tokens.)"

    return answer, response.usage.input_tokens


def main() -> None:
    load_dotenv()  # reads ANTHROPIC_API_KEY from .env
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_pdf()
    system_prompt = build_system_prompt(
        build_document_text(extract_pages(pdf_path))
    )
    client = anthropic.Anthropic()

    print(f"Loaded {pdf_path}. Ask a question (empty line to quit).")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except EOFError:
            break
        if not question:
            break

        try:
            answer, input_tokens = ask(client, system_prompt, question)
        except anthropic.AuthenticationError:
            sys.exit("Invalid API key. Check ANTHROPIC_API_KEY in .env.")
        except anthropic.RateLimitError:
            print("Rate limited. Wait a moment and try again.")
            continue
        except anthropic.APIConnectionError:
            print("Network error. Check your internet connection.")
            continue
        except anthropic.APIStatusError as error:
            print(f"API error {error.status_code}: {error.message}")
            continue

        print(f"\n{answer}")
        print(f"\n[Input tokens: {input_tokens:,}]")


if __name__ == "__main__":
    main()
