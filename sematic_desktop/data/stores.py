"""Lance-backed persistence helpers that wrap foundation-level functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sematic_desktop.foundation.lance import (
    create_doc_table,
    create_metadata_table,
    create_tag_table,
    fetch_metadata_rows,
    list_doc_sources,
    list_tag_pairs,
    search_vectors,
    upsert_metadata_row,
    upsert_vectors,
)

__all__ = ["LanceMetadataStore", "LanceEmbeddingStore"]


class LanceMetadataStore:
    """Persists metadata for each document into a Lance table."""

    def __init__(self, root: Path | str, table_name: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.table_name = table_name
        self.table = create_metadata_table(self.root, self.table_name)
        self._known_sources: set[str] | None = None

    def _normalize_path(self, source_path: Path | str) -> str:
        return str(Path(source_path).expanduser().resolve())

    def has_record(self, source_path: Path | str) -> bool:
        path_str = self._normalize_path(source_path)
        sources = self._load_known_sources()
        return path_str in sources

    def upsert(self, record: dict[str, Any]) -> None:
        """Replace any existing record for ``source_path`` before inserting."""

        record["source_path"] = self._normalize_path(record["source_path"])
        upsert_metadata_row(self.table, record)
        self._known_sources = None

    def fetch_by_paths(self, paths: list[str]) -> dict[str, dict[str, Any]]:
        """Return metadata rows keyed by ``source_path`` for the provided paths."""

        return fetch_metadata_rows(self.table, paths)

    def _load_known_sources(self) -> set[str]:
        if self._known_sources is None:
            arrow_table = self.table.to_arrow(columns=["source_path"])
            if arrow_table.num_rows:
                column = arrow_table.column("source_path")
                self._known_sources = set(str(value) for value in column.to_pylist())
            else:
                self._known_sources = set()
        return self._known_sources


class LanceEmbeddingStore:
    """Persists embeddings for each document/variant combination."""

    def __init__(
        self,
        root: Path | str,
        doc_table_name: str = "emb_doc",
        *,
        tag_table_name: str = "emb_tags",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.doc_table_name = doc_table_name
        self.tag_table_name = tag_table_name
        self.doc_table = create_doc_table(self.root, self.doc_table_name)
        self.tag_table = create_tag_table(self.root, self.tag_table_name)
        self._known_documents: set[str] | None = None
        self._known_tag_pairs: set[tuple[str, str]] | None = None

    def _normalize_path(self, source_path: Path | str) -> str:
        return str(Path(source_path).expanduser().resolve())

    def has_variant(self, source_path: Path | str, variant: str) -> bool:
        normalized = self._normalize_path(source_path)
        if variant == "document":
            docs = self._load_known_documents()
            return normalized in docs
        if variant == "tags":
            pairs = self._load_known_tag_pairs()
            return any(source == normalized for source, _ in pairs)
        raise ValueError(f"Unknown embedding variant '{variant}'")

    def upsert_many(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        doc_records: list[dict[str, Any]] = []
        tag_records: list[dict[str, Any]] = []
        for record in records:
            variant = record.get("variant")
            source_path = self._normalize_path(record["source_path"])
            markdown_path = str(record["markdown_path"])
            vector = record["vector"]
            if variant == "document":
                doc_records.append(
                    {
                        "source_path": source_path,
                        "markdown_path": markdown_path,
                        "vector": vector,
                    },
                )
            elif variant == "tags":
                tag_text = str(record.get("variant_label") or "").strip()
                if not tag_text:
                    continue
                tag_records.append(
                    {
                        "source_path": source_path,
                        "markdown_path": markdown_path,
                        "tag_text": tag_text,
                        "vector": vector,
                    },
                )
        upsert_vectors(
            doc_table=self.doc_table,
            tag_table=self.tag_table,
            doc_records=doc_records,
            tag_records=tag_records,
        )
        if doc_records:
            self._known_documents = None
        if tag_records:
            self._known_tag_pairs = None

    def search(
        self, vector: list[float], *, variant: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        if variant == "document":
            rows = search_vectors(self.doc_table, vector, limit=limit)
            for row in rows:
                row["variant"] = "document"
                row["variant_label"] = None
            return rows
        if variant == "tags":
            rows = search_vectors(self.tag_table, vector, limit=limit)
            for row in rows:
                row["variant"] = "tags"
                row["variant_label"] = row.get("tag_text")
            return rows
        raise ValueError(f"Unknown embedding variant '{variant}'")

    def _load_known_documents(self) -> set[str]:
        if self._known_documents is None:
            self._known_documents = list_doc_sources(self.doc_table)
        return self._known_documents

    def _load_known_tag_pairs(self) -> set[tuple[str, str]]:
        if self._known_tag_pairs is None:
            self._known_tag_pairs = list_tag_pairs(self.tag_table)
        return self._known_tag_pairs
