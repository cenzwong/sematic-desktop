"""Utilities for walking folders and collecting files."""
from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Iterable, Sequence

# File types that are generally useful for semantic search ingestion.
DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".txt",
    ".md",
    ".markdown",
    ".rtf",
    ".pdf",
    ".doc",
    ".docx",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
)


def _normalized_extensions(extensions: Sequence[str] | None) -> set[str]:
    if not extensions:
        return set()
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}


def list_files(folder: Path | str, *, allowed_extensions: Iterable[str] | None = None) -> list[Path]:
    """Return indexed files contained within ``folder``.

    Args:
        folder: Directory to walk recursively.
        allowed_extensions: Optional override for which file types are returned.
            If omitted, ``DEFAULT_EXTENSIONS`` is used.

    Returns:
        Sorted list of file paths underneath ``folder`` filtered by extension.

    Raises:
        ValueError: If ``folder`` does not exist or is not a directory.
    """
    base_path = Path(folder).expanduser()
    if not base_path.exists():
        raise ValueError(f"Folder {base_path} does not exist.")
    if not base_path.is_dir():
        raise ValueError(f"Path {base_path} is not a directory.")

    allowed = (
        _normalized_extensions(tuple(allowed_extensions))
        if allowed_extensions is not None
        else set(DEFAULT_EXTENSIONS)
    )

    pattern = str(base_path / "**" / "*")
    files = []
    for entry in glob(pattern, recursive=True):
        path = Path(entry)
        if path.is_file():
            if not allowed or path.suffix.lower() in allowed:
                files.append(path)

    files.sort()
    return files


__all__ = ["list_files", "DEFAULT_EXTENSIONS"]
