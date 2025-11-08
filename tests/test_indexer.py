"""Integration tests for build_markdown_index Lance metadata support."""
from __future__ import annotations

import lancedb

import pytest

from sematic_desktop.indexer import build_markdown_index
from sematic_desktop.summarizer import MarkdownSummary


class DummyMarkItDown:
    def convert(self, _: str) -> object:  # pragma: no cover - exercised indirectly.
        class Result:
            text_content = "# Title\n\nBody"

        return Result()


class DummySummarizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def summarize(self, markdown_text: str) -> MarkdownSummary:
        self.calls.append(markdown_text)
        return MarkdownSummary(description="desc", tags=["tag"])


class DummyEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, markdown_text: str) -> list[float]:
        self.calls.append(markdown_text)
        return [0.1, 0.2, 0.3]


def test_build_markdown_index_writes_lance_metadata(tmp_path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    (source_dir / "note.txt").write_text("hello world", encoding="utf-8")

    summarizer = DummySummarizer()
    embedding_client = DummyEmbeddingClient()
    outputs = build_markdown_index(
        source_dir,
        output_root=tmp_path / "markdown",
        metadata_root=tmp_path / "metadata",
        allowed_extensions=["txt"],
        markitdown_converter=DummyMarkItDown(),
        docling_converter=None,
        show_progress=False,
        markdown_summarizer=summarizer,
        embedding_client=embedding_client,
    )

    assert len(outputs) == 1
    db = lancedb.connect(str(tmp_path / "metadata" / "docs"))
    table = db.open_table("properties")
    rows = table.to_arrow().to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert row["description"] == "desc"
    assert row["tags"] == ["tag"]
    assert summarizer.calls
    assert embedding_client.calls

    embedding_db = lancedb.connect(str(tmp_path / "metadata" / "docs"))
    embedding_table = embedding_db.open_table("embeddings")
    embedding_rows = embedding_table.to_arrow().to_pylist()
    assert len(embedding_rows) == 2  # one document vector + one tag vector
    document_row = next(row for row in embedding_rows if row["variant"] == "document")
    tags_row = next(
        row for row in embedding_rows if row["variant"] == "tags" and row["variant_label"] == "tag"
    )
    assert document_row["vector"] == pytest.approx([0.1, 0.2, 0.3])
    assert tags_row["vector"] == pytest.approx([0.1, 0.2, 0.3])
