"""Data access layer for Lance-backed storage."""

from .stores import LanceEmbeddingStore, LanceMetadataStore

__all__ = ["LanceEmbeddingStore", "LanceMetadataStore"]
