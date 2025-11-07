"""Core package for sematic-desktop utilities."""

from .embeddings import EmbeddingGemmaClient, EmbeddingGemmaError, LanceMetadataStore
from .indexer import (
    DEFAULT_EXTENSIONS,
    DEFAULT_MARKDOWN_ROOT,
    DEFAULT_METADATA_ROOT,
    build_markdown_index,
    list_files,
)
from .summarizer import MarkdownSummarizer, MarkdownSummary

__all__ = [
    "EmbeddingGemmaClient",
    "EmbeddingGemmaError",
    "DEFAULT_EXTENSIONS",
    "DEFAULT_MARKDOWN_ROOT",
    "DEFAULT_METADATA_ROOT",
    "LanceMetadataStore",
    "MarkdownSummarizer",
    "MarkdownSummary",
    "build_markdown_index",
    "list_files",
]
