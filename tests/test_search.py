"""Tests for the semantic search workflows."""

from __future__ import annotations

from pathlib import Path

from sematic_desktop.data.stores import LanceEmbeddingStore, LanceMetadataStore
from sematic_desktop.services.search import SemanticSearchEngine


class StubEmbeddingClient:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, text: str) -> list[float]:  # pragma: no cover - exercised via engine
        key = text.strip().lower()
        return self.mapping.get(key, self.mapping["default"])


class StubAnswerer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def answer(self, question: str, contexts: list[dict[str, str]]) -> str:
        self.calls.append((question, contexts))
        return f"answer: {question}"


def _write_markdown(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / "markdown" / "docs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_search_engine_supports_all_modes(tmp_path) -> None:
    metadata_store = LanceMetadataStore(tmp_path / "metadata", "docs")
    embedding_store = LanceEmbeddingStore(
        tmp_path / "embeddings", doc_table_name="emb_doc", tag_table_name="emb_tags"
    )
    markdown_path = _write_markdown(tmp_path, "note.md", "Important facts live here.")

    metadata_store.upsert(
        {
            "source_path": str(tmp_path / "docs" / "note.txt"),
            "markdown_path": str(markdown_path),
            "converter": "markitdown",
            "size_bytes": 10,
            "indexed_at": "2024-01-01T00:00:00+00:00",
            "modified_at": "2024-01-01T00:00:00+00:00",
            "file_name": "note.txt",
            "file_extension": ".txt",
            "file_type": "text/plain",
            "description": "A helpful summary.",
            "tags": ["alpha", "beta"],
        },
    )
    embedding_store.upsert_many(
        [
            {
                "source_path": str(tmp_path / "docs" / "note.txt"),
                "markdown_path": str(markdown_path),
                "variant": "document",
                "variant_label": None,
                "vector": [1.0, 0.0],
            },
            {
                "source_path": str(tmp_path / "docs" / "note.txt"),
                "markdown_path": str(markdown_path),
                "variant": "tags",
                "variant_label": "beta",
                "vector": [0.0, 1.0],
            },
        ],
    )

    embedding_client = StubEmbeddingClient(
        {
            "context query": [1.0, 0.0],
            "tag query": [0.0, 1.0],
            "what is note": [1.0, 0.0],
            "default": [1.0, 0.0],
        },
    )
    answerer = StubAnswerer()
    engine = SemanticSearchEngine(
        metadata_store,
        embedding_store,
        embedding_client=embedding_client,
        answerer=answerer,
    )

    context_hits = engine.search_context("context query")
    assert context_hits and context_hits[0].description == "A helpful summary."

    tag_hits = engine.search_tags("tag query")
    assert tag_hits and tag_hits[0].variant == "tags"
    assert tag_hits[0].matched_tag == "beta"

    answer_payload = engine.answer_question("What is note?", top_k=1)
    assert "answer: What is note?" == answer_payload["answer"]
    assert answerer.calls
    assert answer_payload["hits"]
