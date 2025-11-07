"""Placeholder helpers for embedding export via embeddinggemma + Lance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .ollama_client import OllamaClient


@dataclass(slots=True)
class EmbeddingTask:
    """Lightweight input structure before Lance ingestion exists."""

    markdown_path: Path
    chunk_text: str
    metadata: dict[str, Any]


class EmbeddingGemmaClient:
    """Scaffold for embedding generation via ``embeddinggemma:latest``."""

    def __init__(
        self,
        *,
        client: OllamaClient | None = None,
        model: str = "embeddinggemma:latest",
    ) -> None:
        self.client = client or OllamaClient()
        self.model = model

    def embed(self, text: str) -> Sequence[float]:
        """Return embeddings for ``text`` (implementation pending)."""
        raise NotImplementedError(
            "Embedding export to Lance will be implemented in a future iteration.",
        )


def write_embeddings_to_lance(
    tasks: Sequence[EmbeddingTask],
    *,
    output_path: Path,
) -> None:
    """Reserved hook for persisting embeddings in Lance format."""
    raise NotImplementedError("Lance-backed embedding persistence is not implemented yet.")


__all__ = ["EmbeddingGemmaClient", "EmbeddingTask", "write_embeddings_to_lance"]
