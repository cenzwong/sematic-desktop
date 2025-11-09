"""Presentation helpers for CLI + future GUI surfaces."""

from .index_cli import run_indexing_cli
from .search_cli import (
    build_search_engine,
    print_property_examples,
    print_rag_answer,
    print_tag_search,
    query_properties,
    resolve_metadata_folder,
)

__all__ = [
    "build_search_engine",
    "print_property_examples",
    "print_rag_answer",
    "print_tag_search",
    "query_properties",
    "resolve_metadata_folder",
    "run_indexing_cli",
]
