"""Example CLI demonstrating metadata + embedding queries.

This script assumes ``uv run python main.py`` (or another indexing pass)
already produced Lance datasets under ``.semantic_index/metadata``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from sematic_desktop import LanceEmbeddingStore, LanceMetadataStore, SemanticSearchEngine


def _normalize_extension(extension: str | None) -> str | None:
    if not extension:
        return None
    extension = extension.strip().lower()
    if not extension:
        return None
    if not extension.startswith("."):
        extension = f".{extension}"
    return extension


def _resolve_metadata_folder(source_folder: Path, metadata_root: Path | None) -> Path:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query Lance metadata + embeddings built by sematic-desktop.",
    )
    parser.add_argument(
        "--folder",
        default="my_folder",
        help="Source folder that was previously indexed (default: %(default)s).",
    )
    parser.add_argument(
        "--metadata-root",
        default=None,
        help="Override path to the '.semantic_index/metadata' root if needed.",
    )
    parser.add_argument(
        "--property-extension",
        default=".md",
        help="Filter metadata rows by file extension (default: %(default)s).",
    )
    parser.add_argument(
        "--min-size-bytes",
        type=int,
        default=0,
        help="Minimum file size to include in the property query (default: %(default)s).",
    )
    parser.add_argument(
        "--property-limit",
        type=int,
        default=5,
        help="Number of metadata rows to print for the property query (default: %(default)s).",
    )
    parser.add_argument(
        "--tag-query",
        default="project status",
        help="Example semantic tag query (default: %(default)s).",
    )
    parser.add_argument(
        "--tag-limit",
        type=int,
        default=3,
        help="Number of tag matches to show (default: %(default)s).",
    )
    parser.add_argument(
        "--qa-question",
        default="What are the most important updates?",
        help="Question to run through the RAG workflow (default: %(default)s).",
    )
    parser.add_argument(
        "--qa-top-k",
        type=int,
        default=3,
        help="Number of documents to ground RAG answers with (default: %(default)s).",
    )

    args = parser.parse_args()
    folder = Path(args.folder)
    metadata_root = Path(args.metadata_root) if args.metadata_root else None
    metadata_folder = _resolve_metadata_folder(folder, metadata_root)
    metadata_store = LanceMetadataStore(metadata_folder, "properties")
    embedding_store = LanceEmbeddingStore(metadata_folder, "embeddings")
    engine = SemanticSearchEngine(metadata_store, embedding_store)

    property_rows = query_properties(
        metadata_store,
        extension=args.property_extension,
        min_size=args.min_size_bytes,
        limit=args.property_limit,
    )
    print_property_examples(property_rows)
    print()

    print_tag_search(engine, query=args.tag_query, top_k=args.tag_limit)
    print()

    print_rag_answer(engine, question=args.qa_question, top_k=args.qa_top_k)


if __name__ == "__main__":
    main()
