"""Middleware clients that talk to external systems."""

from .embeddings import EmbeddingGemmaClient, EmbeddingGemmaError
from .ollama import OllamaClient, OllamaError
from .routing import ConversionRouter, FileSignals, gather_file_signals
from .summarizer import MarkdownSummarizer, MarkdownSummary

__all__ = [
    "ConversionRouter",
    "EmbeddingGemmaClient",
    "EmbeddingGemmaError",
    "FileSignals",
    "MarkdownSummarizer",
    "MarkdownSummary",
    "OllamaClient",
    "OllamaError",
    "gather_file_signals",
]

