"""CLI helpers for kicking off the indexing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sematic_desktop.services.indexing import build_markdown_index

__all__ = ["run_indexing_cli", "print_index_results"]


def run_indexing_cli(folder: Path | str = "./my_folder", **kwargs) -> list[Path]:
    """Invoke the indexing pipeline and return written markdown paths."""
    return build_markdown_index(folder, **kwargs)


def print_index_results(paths: Iterable[Path]) -> None:
    """Render CLI-friendly output for indexed files."""
    paths = list(paths)
    if not paths:
        print("No markdown files were written.")
        return
    print("Generated markdown artifacts:")
    for path in paths:
        print(path)
