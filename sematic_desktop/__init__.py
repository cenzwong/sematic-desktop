"""Core package for sematic-desktop utilities."""

from .data.stores import LanceEmbeddingStore, LanceMetadataStore
from .middleware.embeddings import EmbeddingGemmaClient, EmbeddingGemmaError
from .middleware.routing import ConversionRouter, FileSignals, gather_file_signals
from .middleware.summarizer import MarkdownSummarizer, MarkdownSummary
from .services.indexing import (
    DEFAULT_EXTENSIONS,
    DEFAULT_MARKDOWN_ROOT,
    DEFAULT_METADATA_ROOT,
    MarkdownIndexService,
    build_markdown_index,
    list_files,
)
from .services.search import ContextAnswerer, SearchHit, SemanticSearchEngine

__all__ = [
    "ContextAnswerer",
    "ConversionRouter",
    "EmbeddingGemmaClient",
    "EmbeddingGemmaError",
    "DEFAULT_EXTENSIONS",
    "DEFAULT_MARKDOWN_ROOT",
    "DEFAULT_METADATA_ROOT",
    "FileSignals",
    "LanceEmbeddingStore",
    "LanceMetadataStore",
    "MarkdownIndexService",
    "MarkdownSummarizer",
    "MarkdownSummary",
    "SearchHit",
    "SemanticSearchEngine",
    "build_markdown_index",
    "gather_file_signals",
    "list_files",
]
