"""Tests for simple_ask.py.

The real Claude API is replaced by a fake client, so the tests are free,
fast, and work without internet or an API key.
"""

from types import SimpleNamespace

import pytest

from simple_ask import (
    ask,
    build_document_text,
    build_system_prompt,
    is_quit,
    read_question,
)


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


@pytest.mark.parametrize("text", ["quit", "exit", "q", "QUIT", "  Exit  ", ""])
def test_is_quit_accepts_quit_words_and_empty(text):
    assert is_quit(text)


@pytest.mark.parametrize("text", ["What is the total?", "quite good", "q1"])
def test_is_quit_rejects_real_questions(text):
    assert not is_quit(text)


def test_read_question_returns_the_question(monkeypatch):
    # monkeypatch swaps input() for a fake, so no keyboard is needed.
    monkeypatch.setattr("builtins.input", lambda prompt: "  What is the total?  ")

    assert read_question() == "What is the total?"


def test_read_question_returns_none_for_quit(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "quit")

    assert read_question() is None


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_read_question_returns_none_on_ctrl_d_or_ctrl_c(monkeypatch, error):
    def fake_input(prompt):
        raise error

    monkeypatch.setattr("builtins.input", fake_input)

    assert read_question() is None
