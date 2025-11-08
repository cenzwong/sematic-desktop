"""Core package for sematic-desktop utilities."""

from .embeddings import (
    EmbeddingGemmaClient,
    EmbeddingGemmaError,
    LanceEmbeddingStore,
    LanceMetadataStore,
)
from .indexer import (
    DEFAULT_EXTENSIONS,
    DEFAULT_MARKDOWN_ROOT,
    DEFAULT_METADATA_ROOT,
    build_markdown_index,
    list_files,
)
from .search import ContextAnswerer, SearchHit, SemanticSearchEngine
from .summarizer import MarkdownSummarizer, MarkdownSummary

__all__ = [
    "ContextAnswerer",
    "EmbeddingGemmaClient",
    "EmbeddingGemmaError",
    "DEFAULT_EXTENSIONS",
    "DEFAULT_MARKDOWN_ROOT",
    "DEFAULT_METADATA_ROOT",
    "LanceEmbeddingStore",
    "LanceMetadataStore",
    "MarkdownSummarizer",
    "MarkdownSummary",
    "SearchHit",
    "SemanticSearchEngine",
    "build_markdown_index",
    "list_files",
]
