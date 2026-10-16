"""Tests for simple_ask.py.

The real Claude API is replaced by a fake client, so the tests are free,
fast, and work without internet or an API key.
"""

import json
from types import SimpleNamespace

import pytest

from simple_ask import (
    NOT_FOUND,
    Reply,
    ask,
    build_document_text,
    build_system_prompt,
    format_reply,
    is_quit,
    parse_reply,
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


def make_fake_client(found=True, answer="Net 30 days.", pages=(3,),
                     stop_reason="end_turn", input_tokens=1234, raw_text=None):
    """A fake client whose reply is the JSON Claude would send back."""
    if raw_text is None:
        raw_text = json.dumps(
            {"found": found, "answer": answer, "pages": list(pages)}
        )
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=raw_text),
        ],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens),
    )
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


# --- building the prompt ---------------------------------------------------

def test_build_document_text_adds_page_markers():
    text = build_document_text(["alpha", "beta"])

    assert text == "--- Page 1 ---\nalpha\n\n--- Page 2 ---\nbeta"


def test_system_prompt_contains_rules_and_document():
    prompt = build_system_prompt("--- Page 1 ---\nInvoice total 42")

    assert "ONLY the document" in prompt
    assert NOT_FOUND in prompt
    assert "Invoice total 42" in prompt


# --- asking Claude -----------------------------------------------------------

def test_ask_sends_question_document_and_json_schema():
    client, messages = make_fake_client()

    ask(client, "SYSTEM WITH DOCUMENT", "What are the payment terms?", {1, 2, 3})

    request = messages.last_request
    assert request["system"] == "SYSTEM WITH DOCUMENT"
    assert request["messages"] == [
        {"role": "user", "content": "What are the payment terms?"}
    ]
    assert request["output_config"]["format"]["type"] == "json_schema"


def test_ask_returns_answer_pages_and_tokens():
    client, _ = make_fake_client(answer="Net 30 days.", pages=[3],
                                 input_tokens=1234)

    reply = ask(client, "system", "question", {1, 2, 3, 4})

    assert reply == Reply("Net 30 days.", [3], 1234)


def test_ask_handles_refusal():
    client, _ = make_fake_client(stop_reason="refusal", raw_text="")

    reply = ask(client, "system", "question", {1})

    assert "declined" in reply.text
    assert reply.pages == []


def test_ask_handles_cut_off_json():
    client, _ = make_fake_client(stop_reason="max_tokens",
                                 raw_text='{"found": true, "answ')

    reply = ask(client, "system", "question", {1})

    assert "incomplete" in reply.text


# --- the "no guessing" rules ------------------------------------------------

def test_not_found_gives_fixed_message():
    raw = json.dumps({"found": False, "answer": "Maybe 42?", "pages": [1]})

    reply = parse_reply(raw, {1, 2}, 100)

    assert reply.text == NOT_FOUND  # Claude's guess is thrown away
    assert reply.pages == []


def test_pages_outside_provided_text_are_dropped():
    raw = json.dumps({"found": True, "answer": "Net 30.", "pages": [3, 9]})

    reply = parse_reply(raw, {2, 3}, 100)

    assert reply.pages == [3]


def test_answer_with_no_valid_page_becomes_not_found():
    raw = json.dumps({"found": True, "answer": "Net 30.", "pages": [9]})

    reply = parse_reply(raw, {2, 3}, 100)

    assert reply.text == NOT_FOUND


def test_pages_are_sorted_and_unique():
    raw = json.dumps({"found": True, "answer": "x", "pages": [3, 1, 3]})

    assert parse_reply(raw, {1, 2, 3}, 100).pages == [1, 3]


# --- showing the answer ------------------------------------------------------

def test_format_reply_shows_single_page():
    assert format_reply(Reply("Net 30 days.", [3], 0)) == (
        "Net 30 days.\n\nSource: page 3"
    )


def test_format_reply_shows_several_pages():
    assert format_reply(Reply("Answer.", [1, 4], 0)).endswith(
        "Source: pages 1, 4"
    )


def test_format_reply_not_found_has_no_source():
    assert format_reply(Reply(NOT_FOUND, [], 0)) == NOT_FOUND


# --- quitting ----------------------------------------------------------------

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
