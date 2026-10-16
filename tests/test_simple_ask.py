"""Tests for simple_ask.py.

The real Claude API is replaced by a fake client, so the tests are free,
fast, and work without internet or an API key.
"""

from types import SimpleNamespace

from simple_ask import ask, build_document_text, build_system_prompt


class FakeMessages:
    """Records the request and returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return self.response


def make_fake_client(text="The total is 42.", stop_reason="end_turn",
                     input_tokens=1234):
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens),
    )
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_build_document_text_adds_page_markers():
    text = build_document_text(["alpha", "beta"])

    assert text == "--- Page 1 ---\nalpha\n\n--- Page 2 ---\nbeta"


def test_system_prompt_contains_instructions_and_document():
    prompt = build_system_prompt("--- Page 1 ---\nInvoice total 42")

    assert "Use only the document" in prompt
    assert "Invoice total 42" in prompt


def test_ask_sends_question_and_document():
    client, messages = make_fake_client()

    ask(client, "SYSTEM WITH DOCUMENT", "What is the total?")

    request = messages.last_request
    assert request["system"] == "SYSTEM WITH DOCUMENT"
    assert request["messages"] == [
        {"role": "user", "content": "What is the total?"}
    ]


def test_ask_returns_text_and_input_tokens():
    client, _ = make_fake_client(text="The total is 42.", input_tokens=1234)

    answer, input_tokens = ask(client, "system", "question")

    assert answer == "The total is 42."  # thinking block is skipped
    assert input_tokens == 1234


def test_ask_handles_refusal():
    client, _ = make_fake_client(text="", stop_reason="refusal")

    answer, _ = ask(client, "system", "question")

    assert "declined" in answer
