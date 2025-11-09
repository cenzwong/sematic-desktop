"""Low-level helpers for connecting to Lance tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import lancedb
import numpy as np
import pyarrow as pa

LanceMetadataTable = Any
LanceDocTable = Any
LanceTagTable = Any


def _connect(root: Path | str):
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(path))


def create_metadata_table(root: Path | str, table_name: str) -> LanceMetadataTable:
    """Return a Lance table for metadata, creating it if needed."""

    db = _connect(root)
    schema = pa.schema(
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
    if table_name in db.table_names():
        return db.open_table(table_name)
    return db.create_table(table_name, schema=schema)


def create_doc_table(root: Path | str, table_name: str) -> LanceDocTable:
    """Return a Lance table for document embeddings."""

    schema = pa.schema(
        [
            pa.field("source_path", pa.string()),
            pa.field("markdown_path", pa.string()),
            pa.field("vector", pa.list_(pa.float32())),
        ]
    )
    return _create_or_upgrade(root, table_name, schema)


def create_tag_table(root: Path | str, table_name: str) -> LanceTagTable:
    """Return a Lance table for tag embeddings."""

    schema = pa.schema(
        [
            pa.field("source_path", pa.string()),
            pa.field("markdown_path", pa.string()),
            pa.field("tag_text", pa.string()),
            pa.field("vector", pa.list_(pa.float32())),
        ]
    )
    return _create_or_upgrade(root, table_name, schema)


def _create_or_upgrade(root: Path | str, table_name: str, schema: pa.Schema):
    db = _connect(root)
    if table_name in db.table_names():
        table = db.open_table(table_name)
        current_fields = list(table.schema.names)
        expected_fields = [field.name for field in schema]
        if current_fields != expected_fields:
            rows = table.to_arrow().to_pylist()
            db.drop_table(table_name)
            table = db.create_table(table_name, schema=schema)
            if rows:
                table.add(rows)
        return table
    return db.create_table(table_name, schema=schema)


def upsert_metadata_row(table: LanceMetadataTable, record: dict[str, Any]) -> None:
    """Insert or replace a metadata row."""

    normalized = Path(record["source_path"]).expanduser().resolve()
    record["source_path"] = str(normalized)
    escaped = str(record["source_path"]).replace("'", "''")
    table.delete(where=f"source_path = '{escaped}'")
    table.add([record])


def upsert_vectors(
    *,
    doc_table: LanceDocTable,
    tag_table: LanceTagTable,
    doc_records: list[dict[str, Any]],
    tag_records: list[dict[str, Any]],
) -> None:
    """Insert or replace document/tag embeddings."""

    for record in doc_records:
        source = str(Path(record["source_path"]).expanduser().resolve())
        record["source_path"] = source
        sanitized = source.replace("'", "''")
        doc_table.delete(where=f"source_path = '{sanitized}'")
    if doc_records:
        doc_table.add(doc_records)

    for record in tag_records:
        source = str(Path(record["source_path"]).expanduser().resolve())
        record["source_path"] = source
        tag = record["tag_text"].replace("'", "''")
        sanitized = source.replace("'", "''")
        where = f"source_path = '{sanitized}' AND tag_text = '{tag}'"
        tag_table.delete(where=where)
    if tag_records:
        tag_table.add(tag_records)


def delete_doc_vector(table: LanceDocTable, source_path: Path | str) -> None:
    """Remove the document vector for ``source_path``."""

    normalized = str(Path(source_path).expanduser().resolve()).replace("'", "''")
    table.delete(where=f"source_path = '{normalized}'")


def delete_tag_vector(
    table: LanceTagTable, source_path: Path | str, tag_text: str
) -> None:
    """Remove the tag vector for ``source_path`` + ``tag_text``."""

    normalized = str(Path(source_path).expanduser().resolve()).replace("'", "''")
    tag = tag_text.replace("'", "''")
    table.delete(where=f"source_path = '{normalized}' AND tag_text = '{tag}'")


def fetch_metadata_rows(
    table: LanceMetadataTable, paths: list[str]
) -> dict[str, dict[str, Any]]:
    """Return metadata rows keyed by ``source_path``."""

    if not paths:
        return {}
    normalized = {str(Path(path).expanduser().resolve()) for path in paths}
    arrow_table = table.to_arrow()
    rows = arrow_table.to_pylist()
    return {row["source_path"]: row for row in rows if row["source_path"] in normalized}


def list_doc_sources(table: LanceDocTable) -> set[str]:
    """Return the normalized source paths that have doc embeddings."""

    arrow_table = table.to_arrow(columns=["source_path"])
    if not arrow_table.num_rows:
        return set()
    column = arrow_table.column("source_path").to_pylist()
    return {str(Path(value).expanduser().resolve()) for value in column}


def list_tag_pairs(table: LanceTagTable) -> set[tuple[str, str]]:
    """Return the normalized source/tag pairs in the tag table."""

    arrow_table = table.to_arrow(columns=["source_path", "tag_text"])
    if not arrow_table.num_rows:
        return set()
    sources = arrow_table.column("source_path").to_pylist()
    tags = arrow_table.column("tag_text").to_pylist()
    return {
        (str(Path(source).expanduser().resolve()), str(tag))
        for source, tag in zip(sources, tags, strict=False)
    }


def search_vectors(
    table, vector: list[float], *, limit: int = 5
) -> list[dict[str, Any]]:
    """Return rows sorted by cosine distance relative to ``vector``."""

    if not vector:
        return []
    query = np.asarray(vector, dtype="float32")
    if query.ndim != 1:
        raise ValueError("Query vector must be one-dimensional.")
    rows = table.to_arrow().to_pylist()
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
