"""Unit tests for the EmbeddingGemmaClient helper."""

from __future__ import annotations

import json

import pytest

from sematic_desktop.middleware.embeddings import EmbeddingGemmaClient


def test_embedding_client_parses_vector_from_response() -> None:
    captured: dict[str, str] = {}

    def transport(body: bytes) -> bytes:
        captured["body"] = body.decode("utf-8")
        return b'{"embedding": [1, 2.5, 3]}'

    client = EmbeddingGemmaClient(transport=transport)
    vector = client.embed("hello world")

    assert vector == [1.0, 2.5, 3.0]
    payload = json.loads(captured["body"])
    assert payload["model"] == "embeddinggemma:latest"
    assert payload["prompt"] == "hello world"


def test_embedding_client_rejects_empty_text() -> None:
    client = EmbeddingGemmaClient(transport=lambda _: b"{}")
    with pytest.raises(ValueError):
        client.embed("   ")
