"""Example CLI demonstrating metadata + embedding queries."""

from __future__ import annotations

import argparse
from pathlib import Path

from sematic_desktop.presentation.search_cli import (
    build_search_engine,
    print_property_examples,
    print_rag_answer,
    print_tag_search,
    query_properties,
    resolve_metadata_folder,
)


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
    metadata_folder = resolve_metadata_folder(folder, metadata_root)
    engine = build_search_engine(metadata_folder)

    metadata_store = engine.metadata_store
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
