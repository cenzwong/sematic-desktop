"""Unit tests for the MarkdownSummarizer integration."""

from __future__ import annotations

from sematic_desktop.middleware.summarizer import MarkdownSummarizer


class StubOllamaClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[tuple[str, str]] = []

    def generate(self, model: str, prompt: str) -> str:  # pragma: no cover - exercised indirectly
        self.requests.append((model, prompt))
        return self.response


def test_markdown_summarizer_normalizes_tags() -> None:
    response = '{"description": "Doc summary", "tags": ["Alpha", "Beta", "alpha"]}'
    client = StubOllamaClient(response)
    summarizer = MarkdownSummarizer(client=client)

    summary = summarizer.summarize("# Sample Heading")

    assert summary.description == "Doc summary"
    assert summary.tags == ["alpha", "beta"]
    assert client.requests  # ensures the client was invoked


def test_markdown_summarizer_extracts_json_from_wrapped_output() -> None:
    response = 'Answer:\n{\n"summary": "Terse", "tags": "one,two"\n}\nDone.'
    client = StubOllamaClient(response)
    summarizer = MarkdownSummarizer(client=client)

    summary = summarizer.summarize("Content goes here.")

    assert summary.description == "Terse"
    assert summary.tags == ["one", "two"]
