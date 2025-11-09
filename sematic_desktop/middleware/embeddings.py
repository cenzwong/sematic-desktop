"""Embedding helpers backed by Ollama's embeddinggemma model."""
from __future__ import annotations

from typing import Callable

from sematic_desktop.foundation.remote_embeddings import request_embedding_vector

__all__ = ["EmbeddingGemmaClient", "EmbeddingGemmaError"]


class EmbeddingGemmaError(RuntimeError):
    """Raised when embedding requests to Ollama fail."""


class EmbeddingGemmaClient:
    """Calls Ollama's local HTTP API to generate embeddings via embeddinggemma."""

    def __init__(
        self,
        *,
        model: str = "embeddinggemma:latest",
        endpoint: str = "http://127.0.0.1:11434/api/embeddings",
        max_chars: int = 4_000,
        timeout: float = 120.0,
        transport: Callable[[bytes], bytes] | None = None,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.max_chars = max_chars
        self.timeout = timeout
        self.transport = transport

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``."""
        prompt = text.strip()
        if not prompt:
            raise ValueError("Cannot embed empty text.")
        if len(prompt) > self.max_chars:
            prompt = prompt[: self.max_chars]

        payload = {"model": self.model, "prompt": prompt}
        try:
            return request_embedding_vector(
                payload,
                endpoint=self.endpoint,
                timeout=self.timeout,
                transport=self.transport,
            )
        except Exception as exc:  # pragma: no cover - best effort.
            raise EmbeddingGemmaError(str(exc)) from exc
