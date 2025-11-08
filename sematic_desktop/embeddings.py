"""Embedding + Lance storage helpers powered by Ollama."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib import error, request

import lancedb
import numpy as np
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
        ]
    )


def _embedding_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source_path", pa.string()),
            pa.field("markdown_path", pa.string()),
            pa.field("variant", pa.string()),  # document, tags, etc.
            pa.field("vector", pa.list_(pa.float32())),
        ]
    )


class LanceMetadataStore:
    """Persists metadata for each document into a Lance table."""

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

    def _normalize_path(self, source_path: Path | str) -> str:
        return str(Path(source_path).expanduser().resolve())

    def _escape_value(self, value: str) -> str:
        return value.replace("'", "''")

    def has_record(self, source_path: Path | str) -> bool:
        path_str = self._normalize_path(source_path)
        sources = self._load_known_sources()
        return path_str in sources

    def upsert(self, record: dict[str, Any]) -> None:
        """Replace any existing record for ``source_path`` before inserting."""
        source_path = self._normalize_path(record["source_path"])
        record["source_path"] = source_path
        self._table.delete(where=f"source_path = '{self._escape_value(source_path)}'")
        self._known_sources = None
        self._table.add([record])

    def fetch_by_paths(self, paths: list[str]) -> dict[str, dict[str, Any]]:
        """Return metadata rows keyed by ``source_path`` for the provided paths."""
        if not paths:
            return {}
        normalized = {self._normalize_path(path) for path in paths}
        arrow_table = self._table.to_arrow()
        rows = arrow_table.to_pylist()
        return {row["source_path"]: row for row in rows if row["source_path"] in normalized}

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


class LanceEmbeddingStore:
    """Persists embeddings for each document/variant combination."""

    def __init__(self, root: Path | str, table_name: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.table_name = table_name
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.root))
        self._table = self._ensure_table()
        self._known_pairs: set[tuple[str, str]] | None = None

    def _ensure_table(self):
        if self.table_name in self._db.table_names():
            return self._db.open_table(self.table_name)
        return self._db.create_table(self.table_name, schema=_embedding_schema())

    def _normalize_path(self, source_path: Path | str) -> str:
        return str(Path(source_path).expanduser().resolve())

    def _escape_value(self, value: str) -> str:
        return value.replace("'", "''")

    def has_variant(self, source_path: Path | str, variant: str) -> bool:
        key = (self._normalize_path(source_path), variant)
        pairs = self._load_known_pairs()
        return key in pairs

    def upsert_many(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        for record in records:
            source_path = self._normalize_path(record["source_path"])
            variant = record["variant"]
            record["source_path"] = source_path
            self._table.delete(
                where=(
                    f"source_path = '{self._escape_value(source_path)}' "
                    f"AND variant = '{self._escape_value(variant)}'"
                ),
            )
        self._known_pairs = None
        self._table.add(records)

    def search(self, vector: list[float], *, variant: str, limit: int = 5) -> list[dict[str, Any]]:
        if not vector:
            return []
        query = np.asarray(vector, dtype="float32")
        if query.ndim != 1:
            raise ValueError("Query vector must be one-dimensional.")
        arrow_table = self._table.to_arrow()
        rows = [row for row in arrow_table.to_pylist() if row["variant"] == variant]
        if not rows:
            return []

        def cosine_distance(other: Sequence[float]) -> float:
            other_vec = np.asarray(other, dtype="float32")
            denom = np.linalg.norm(query) * np.linalg.norm(other_vec)
            if denom == 0:
                return float("inf")
            similarity = float(np.dot(query, other_vec) / denom)
            return 1.0 - similarity

        scored_rows: list[tuple[dict[str, Any], float]] = []
        for row in rows:
            distance = cosine_distance(row["vector"])
            row["_distance"] = distance
            scored_rows.append((row, distance))

        scored_rows.sort(key=lambda item: item[1])
        return [row for row, _ in scored_rows[:limit]]

    def _load_known_pairs(self) -> set[tuple[str, str]]:
        if self._known_pairs is None:
            arrow_table = self._table.to_arrow(columns=["source_path", "variant"])
            if arrow_table.num_rows:
                source_col = arrow_table.column("source_path").to_pylist()
                variant_col = arrow_table.column("variant").to_pylist()
                self._known_pairs = {
                    (str(source), str(variant))
                    for source, variant in zip(source_col, variant_col, strict=False)
                }
            else:
                self._known_pairs = set()
        return self._known_pairs

    @property
    def table(self):
        return self._table


__all__ = [
    "EmbeddingGemmaClient",
    "EmbeddingGemmaError",
    "LanceEmbeddingStore",
    "LanceMetadataStore",
]
