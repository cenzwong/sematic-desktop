"""Embedding + Lance storage helpers powered by Ollama."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)

JsonBytes = bytes


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
        transport: Callable[[JsonBytes], JsonBytes] | None = None,
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
        body = json.dumps(payload).encode("utf-8")
        raw = self._send_request(body)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - best effort.
            raise EmbeddingGemmaError("Embedding response was not valid JSON.") from exc

        embedding = self._extract_embedding_vector(data)
        if embedding is None:
            raise EmbeddingGemmaError("Embedding response did not include a vector.")
        return [float(value) for value in embedding]

    def _send_request(self, body: JsonBytes) -> JsonBytes:
        if self.transport is not None:
            return self.transport(body)

        req = request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return response.read()
        except error.URLError as exc:  # pragma: no cover - network failures are rare.
            raise EmbeddingGemmaError(f"Failed to contact Ollama embeddings API: {exc}") from exc

    @staticmethod
    def _extract_embedding_vector(payload: dict[str, Any]) -> list[float] | None:
        candidate = payload.get("embedding")
        if isinstance(candidate, list):
            return candidate  # type: ignore[return-value]
        data = payload.get("data")
        if isinstance(data, list) and data:
            inner = data[0]
            if isinstance(inner, dict):
                embedding = inner.get("embedding")
                if isinstance(embedding, list):
                    return embedding  # type: ignore[return-value]
        return None


def _metadata_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source_path", pa.string()),
            pa.field("markdown_path", pa.string()),
            pa.field("converter", pa.string()),
            pa.field("size_bytes", pa.int64()),
            pa.field("indexed_at", pa.string()),
            pa.field("modified_at", pa.string()),
            pa.field("file_name", pa.string()),
            pa.field("file_extension", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("description", pa.string()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("embedding", pa.list_(pa.float32())),
        ]
    )


class LanceMetadataStore:
    """Persists metadata + embeddings into a Lance table."""

    def __init__(self, root: Path | str, table_name: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.table_name = table_name
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.root))
        self._table = self._ensure_table()
        self._known_sources: set[str] | None = None

    def _ensure_table(self):
        if self.table_name in self._db.table_names():
            return self._db.open_table(self.table_name)
        return self._db.create_table(self.table_name, schema=_metadata_schema())

    def has_record(self, source_path: Path | str) -> bool:
        path_str = str(Path(source_path).expanduser().resolve())
        sources = self._load_known_sources()
        return path_str in sources

    def append(self, record: dict[str, Any]) -> None:
        """Append (or effectively upsert) a record for ``source_path``."""
        source_path = record.get("source_path")
        if isinstance(source_path, str) and self._known_sources is not None:
            self._known_sources.add(source_path)
        self._table.add([record])

    def _load_known_sources(self) -> set[str]:
        if self._known_sources is None:
            arrow_table = self._table.to_arrow(columns=["source_path"])
            if arrow_table.num_rows:
                column = arrow_table.column("source_path")
                self._known_sources = set(str(value) for value in column.to_pylist())
            else:
                self._known_sources = set()
        return self._known_sources

    @property
    def table(self):
        return self._table


__all__ = ["EmbeddingGemmaClient", "EmbeddingGemmaError", "LanceMetadataStore"]
