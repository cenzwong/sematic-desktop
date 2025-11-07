"""Core package for sematic-desktop utilities."""

from .app import get_greeting, run
from .indexer import DEFAULT_EXTENSIONS, list_files

__all__ = ["DEFAULT_EXTENSIONS", "get_greeting", "list_files", "run"]
