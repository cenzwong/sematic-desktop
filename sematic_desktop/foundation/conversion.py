"""Low-level helpers for running MarkItDown and Docling conversions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:  # pragma: no cover - exercised via dependency injection in tests.
    from markitdown import MarkItDown as _MarkItDownClass
except ImportError:  # pragma: no cover - dependency is optional.
    _MarkItDownClass = None

try:  # pragma: no cover - exercised via dependency injection in tests.
    from docling.document_converter import DocumentConverter as _DoclingConverterClass
except ImportError:  # pragma: no cover - dependency is optional.
    _DoclingConverterClass = None


@dataclass(slots=True)
class ConversionPlan:
    """Defines the ordered converters to attempt for a file."""

    ordered_converters: Sequence[str]


def build_conversion_plan(preferred: Sequence[str] | None = None) -> ConversionPlan:
    """Return a plan ensuring both converters are represented once."""

    plan: list[str] = []
    for name in preferred or []:
        if name in {"markitdown", "docling"} and name not in plan:
            plan.append(name)
    for fallback in ("markitdown", "docling"):
        if fallback not in plan:
            plan.append(fallback)
    return ConversionPlan(tuple(plan))


def extract_markdown_from_markitdown(result: Any) -> str | None:
    """Pull markdown or text content from MarkItDown outputs."""
    for attr in ("text_content", "markdown", "text"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(result, str) and result.strip():
        return result
    return None


def extract_markdown_from_docling(result: Any) -> str | None:
    """Pull markdown or document exports from Docling."""
    document = getattr(result, "document", None)
    if document and hasattr(document, "export_to_markdown"):
        markdown = document.export_to_markdown()
        if isinstance(markdown, str) and markdown.strip():
            return markdown
    if hasattr(result, "export_to_markdown"):
        markdown = result.export_to_markdown()
        if isinstance(markdown, str) and markdown.strip():
            return markdown
    return None


def convert_with_markitdown(source_path: Path, *, override: Any | None = None) -> str | None:
    """Invoke MarkItDown for ``source_path`` and return markdown text."""
    converter = override or (_MarkItDownClass() if _MarkItDownClass is not None else None)
    if converter is None:
        return None
    result = converter.convert(str(source_path))
    return extract_markdown_from_markitdown(result)


def convert_with_docling(source_path: Path, *, override: Any | None = None) -> str | None:
    """Invoke Docling for ``source_path`` and return markdown text."""
    converter = override or (
        _DoclingConverterClass() if _DoclingConverterClass is not None else None
    )
    if converter is None:
        return None
    result = converter.convert(str(source_path))
    return extract_markdown_from_docling(result)
