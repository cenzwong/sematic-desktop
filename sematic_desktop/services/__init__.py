"""Business logic layer for indexing and search workflows."""

from .indexing import (
    DEFAULT_EXTENSIONS,
    DEFAULT_MARKDOWN_ROOT,
    DEFAULT_METADATA_ROOT,
    MarkdownIndexService,
    build_markdown_index,
    list_files,
)
from .search import ContextAnswerer, SearchHit, SemanticSearchEngine

__all__ = [
    "ContextAnswerer",
    "DEFAULT_EXTENSIONS",
    "DEFAULT_MARKDOWN_ROOT",
    "DEFAULT_METADATA_ROOT",
    "MarkdownIndexService",
    "SearchHit",
    "SemanticSearchEngine",
    "build_markdown_index",
    "list_files",
]

