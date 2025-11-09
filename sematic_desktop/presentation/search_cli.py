"""Helpers for the CLI that explores Lance-powered search results."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sematic_desktop.data import LanceEmbeddingStore, LanceMetadataStore
from sematic_desktop.services.search import SemanticSearchEngine

__all__ = [
    "build_search_engine",
    "print_property_examples",
    "print_rag_answer",
    "print_tag_search",
    "query_properties",
    "resolve_metadata_folder",
]


def _normalize_extension(extension: str | None) -> str | None:
    if not extension:
        return None
    extension = extension.strip().lower()
    if not extension:
        return None
    if not extension.startswith("."):
        extension = f".{extension}"
    return extension


def resolve_metadata_folder(source_folder: Path, metadata_root: Path | None) -> Path:
    base = source_folder.expanduser().resolve()
    computed_root = (
        metadata_root.expanduser().resolve()
        if metadata_root is not None
        else base.parent / ".semantic_index" / "metadata"
    )
    folder = computed_root / base.name
    if not folder.exists():
        raise FileNotFoundError(
            f"No Lance datasets were found at {folder}. "
            "Run 'uv run python main.py' to build the index first.",
        )
    return folder


def query_properties(
    metadata_store: LanceMetadataStore,
    *,
    extension: str | None,
    min_size: int,
    limit: int,
) -> list[dict[str, str | int | list[str]]]:
    extension = _normalize_extension(extension)
    arrow_rows = metadata_store.table.to_arrow().to_pylist()
    matches = []
    for row in arrow_rows:
        if extension and row.get("file_extension") != extension:
            continue
        if int(row.get("size_bytes") or 0) < min_size:
            continue
        matches.append(row)
    matches.sort(key=lambda item: int(item.get("size_bytes") or 0), reverse=True)
    return matches[:limit]


def print_property_examples(rows: Iterable[dict[str, str | int | list[str]]]) -> None:
    rows = list(rows)
    if not rows:
        print("No metadata rows matched the property filters.")
        return
    print("Top matches by metadata properties:")
    for row in rows:
        tags = row.get("tags") or []
        tag_text = ", ".join(str(tag) for tag in tags) if tags else "-"
        print(
            f"- {row.get('file_name')} ({row.get('size_bytes')} bytes) | "
            f"type={row.get('file_type')} | tags={tag_text}",
        )


def print_tag_search(engine: SemanticSearchEngine, *, query: str, top_k: int) -> None:
    query = query.strip()
    if not query:
        print("Tag search skipped: empty query.")
        return
    hits = engine.search_tags(query, top_k=top_k)
    if not hits:
        print(f"No semantic tag matches found for '{query}'.")
        return
    print(f"Semantic tag matches for '{query}':")
    for hit in hits:
        tag = hit.matched_tag or "n/a"
        print(f"- {hit.source_path} | tag={tag} | score={hit.score:.3f}")


def print_rag_answer(engine: SemanticSearchEngine, *, question: str, top_k: int) -> None:
    question = question.strip()
    if not question:
        print("RAG example skipped: empty question.")
        return
    payload = engine.answer_question(question, top_k=top_k)
    answer = payload.get("answer", "")
    hits = payload.get("hits", [])
    print(f"RAG answer for '{question}':\n{answer}\n")
    if not hits:
        print("No supporting documents were returned.")
        return
    print("Supporting documents:")
    for hit in hits:
        print(f"- {hit.source_path} | score={hit.score:.3f}")


def build_search_engine(metadata_folder: Path) -> SemanticSearchEngine:
    metadata_store = LanceMetadataStore(metadata_folder, "properties")
    embedding_store = LanceEmbeddingStore(metadata_folder, doc_table_name="emb_doc", tag_table_name="emb_tags")
    return SemanticSearchEngine(metadata_store, embedding_store)
