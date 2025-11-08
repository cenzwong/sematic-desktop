"""Utilities for walking folders and collecting files."""
from __future__ import annotations

from datetime import datetime, timezone
from glob import glob
import logging
import mimetypes
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # pragma: no cover - tqdm is optional during tests.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable.
    def tqdm(iterable, **_: Any):  # type: ignore[override]
        return iterable

from .embeddings import EmbeddingGemmaClient, LanceEmbeddingStore, LanceMetadataStore
from .routing import ConversionRouter, gather_file_signals
from .summarizer import MarkdownSummarizer

try:  # pragma: no cover - exercised via dependency injection in tests.
    from markitdown import MarkItDown as _MarkItDownClass
except ImportError:  # pragma: no cover - dependency is optional at test time.
    _MarkItDownClass = None

try:  # pragma: no cover - exercised via dependency injection in tests.
    from docling.document_converter import DocumentConverter as _DoclingConverterClass
except ImportError:  # pragma: no cover - dependency is optional at test time.
    _DoclingConverterClass = None

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

DEFAULT_MARKDOWN_ROOT = ".semantic_index/markdown"
DEFAULT_METADATA_ROOT = ".semantic_index/metadata"

_MARKITDOWN_INSTANCE: Any | None = None
_DOCLING_INSTANCE: Any | None = None
_MARKDOWN_SUMMARIZER_INSTANCE: MarkdownSummarizer | None = None
_EMBEDDING_CLIENT_INSTANCE: EmbeddingGemmaClient | None = None
logger = logging.getLogger(__name__)


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
    base_path = Path(folder).expanduser().resolve()
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


def _extract_markdown_from_markitdown(result: Any) -> str | None:
    for attr in ("text_content", "markdown", "text"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(result, str) and result.strip():
        return result
    return None


def _extract_markdown_from_docling(result: Any) -> str | None:
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


def _get_markitdown_converter(markitdown_converter: Any | None = None) -> Any | None:
    if markitdown_converter is not None:
        return markitdown_converter
    global _MARKITDOWN_INSTANCE
    if _MARKITDOWN_INSTANCE is None and _MarkItDownClass is not None:
        _MARKITDOWN_INSTANCE = _MarkItDownClass()
    return _MARKITDOWN_INSTANCE


def _get_docling_converter(docling_converter: Any | None = None) -> Any | None:
    if docling_converter is not None:
        return docling_converter
    global _DOCLING_INSTANCE
    if _DOCLING_INSTANCE is None and _DoclingConverterClass is not None:
        _DOCLING_INSTANCE = _DoclingConverterClass()
    return _DOCLING_INSTANCE


def _get_markdown_summarizer(markdown_summarizer: MarkdownSummarizer | None = None) -> MarkdownSummarizer | None:
    if markdown_summarizer is not None:
        return markdown_summarizer
    global _MARKDOWN_SUMMARIZER_INSTANCE
    if _MARKDOWN_SUMMARIZER_INSTANCE is None:
        try:
            _MARKDOWN_SUMMARIZER_INSTANCE = MarkdownSummarizer()
        except Exception as exc:  # pragma: no cover - fallback when Ollama fails to init.
            logger.warning("Disabling markdown summaries: %s", exc)
            _MARKDOWN_SUMMARIZER_INSTANCE = None
    return _MARKDOWN_SUMMARIZER_INSTANCE


def _get_embedding_client(embedding_client: EmbeddingGemmaClient | None = None) -> EmbeddingGemmaClient | None:
    if embedding_client is not None:
        return embedding_client
    global _EMBEDDING_CLIENT_INSTANCE
    if _EMBEDDING_CLIENT_INSTANCE is None:
        try:
            _EMBEDDING_CLIENT_INSTANCE = EmbeddingGemmaClient()
        except Exception as exc:  # pragma: no cover - best effort integration.
            logger.warning("Disabling embeddings: %s", exc)
            _EMBEDDING_CLIENT_INSTANCE = None
    return _EMBEDDING_CLIENT_INSTANCE


def _convert_to_markdown(
    source_path: Path,
    *,
    markitdown_converter: Any | None = None,
    docling_converter: Any | None = None,
    router: ConversionRouter | None = None,
) -> tuple[str, str]:
    errors: list[str] = []
    router = router or ConversionRouter()
    signals = gather_file_signals(
        source_path,
        historical_success=router.historical_success_for(source_path.suffix.lower()),
    )

    preferred_order = router.plan_order(signals)
    order: list[str] = []
    for candidate in preferred_order + ["markitdown", "docling"]:
        if candidate in {"markitdown", "docling"} and candidate not in order:
            order.append(candidate)

    best_attempt: tuple[str, str, float] | None = None

    for converter_name in order:
        if converter_name == "markitdown":
            converter = _get_markitdown_converter(markitdown_converter)
            extractor = _extract_markdown_from_markitdown
        else:
            converter = _get_docling_converter(docling_converter)
            extractor = _extract_markdown_from_docling

        if converter is None:
            errors.append(f"{converter_name}: converter unavailable")
            continue

        try:
            result = converter.convert(str(source_path))
        except Exception as exc:  # pragma: no cover - fallback scenario.
            errors.append(f"{converter_name}: {exc}")
            router.record_outcome(
                signals,
                converter_name,
                success=False,
                quality=0.0,
                error=str(exc),
            )
            continue

        markdown = extractor(result)
        if not markdown:
            message = f"{converter_name}: returned no text"
            errors.append(message)
            router.record_outcome(
                signals,
                converter_name,
                success=False,
                quality=0.0,
                error=message,
            )
            continue

        quality = router.score_markdown(markdown, signals)
        router.record_outcome(
            signals,
            converter_name,
            success=True,
            quality=quality,
        )
        if router.is_quality_acceptable(quality, signals):
            return markdown, converter_name

        errors.append(f"{converter_name}: below quality threshold ({quality:.2f})")
        if best_attempt is None or quality > best_attempt[2]:
            best_attempt = (markdown, converter_name, quality)

    if best_attempt:
        markdown, converter_name, _ = best_attempt
        return markdown, converter_name

    if not errors:
        errors.append("no markdown converter available")
    raise RuntimeError(f"Unable to convert {source_path} to markdown ({'; '.join(errors)})")


def build_markdown_index(
    folder: Path | str,
    *,
    output_root: Path | str | None = None,
    metadata_root: Path | str | None = None,
    allowed_extensions: Iterable[str] | None = None,
    markitdown_converter: Any | None = None,
    docling_converter: Any | None = None,
    router: ConversionRouter | None = None,
    show_progress: bool = True,
    markdown_summarizer: MarkdownSummarizer | None = None,
    enable_markdown_summaries: bool = True,
    embedding_client: EmbeddingGemmaClient | None = None,
    enable_embeddings: bool = True,
) -> list[Path]:
    """Convert files beneath ``folder`` into Markdown artifacts.

    Args:
        folder: Directory whose contents should be indexed.
        output_root: Root directory where `.semantic_index/markdown/<folder>` lives.
            Defaults to the folder's parent joined with ``DEFAULT_MARKDOWN_ROOT``.
        metadata_root: Root directory where `.semantic_index/metadata/<folder>` lives.
            Each folder contains ``properties.lance`` and ``embeddings.lance`` tables.
        allowed_extensions: Optional filter for `list_files`. Provide an empty iterable
            to include every file when desired.
        markitdown_converter: Optional pre-built MarkItDown instance (used for testing).
        docling_converter: Optional Docling converter used as a fallback.
        router: Converter router that decides which engine to invoke and when to retry.
        show_progress: When True, display a progress bar for pending files using tqdm.
        markdown_summarizer: Optional Ollama-backed summarizer override.
        enable_markdown_summaries: When True, persist description + tags using Gemma.
        embedding_client: Optional embedding helper override (defaults to EmbeddingGemma).
        enable_embeddings: When True, store embedding vectors alongside metadata.

    Returns:
        Sorted list of Markdown file paths written to disk.
    """
    base_path = Path(folder).expanduser().resolve()
    if not base_path.exists():
        raise ValueError(f"Folder {base_path} does not exist.")
    if not base_path.is_dir():
        raise ValueError(f"Path {base_path} is not a directory.")

    output_root_path = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (base_path.parent / DEFAULT_MARKDOWN_ROOT)
    )
    target_root = output_root_path / base_path.name
    target_root.mkdir(parents=True, exist_ok=True)

    metadata_root_path = (
        Path(metadata_root).expanduser().resolve()
        if metadata_root is not None
        else (base_path.parent / DEFAULT_METADATA_ROOT)
    )
    metadata_root_path.mkdir(parents=True, exist_ok=True)
    metadata_folder = metadata_root_path / base_path.name
    metadata_folder.mkdir(parents=True, exist_ok=True)
    metadata_store = LanceMetadataStore(metadata_folder, "properties")
    embedding_store = LanceEmbeddingStore(metadata_folder, "embeddings")

    files_to_index = list_files(
        base_path,
        allowed_extensions=allowed_extensions if allowed_extensions is not None else [],
    )
    summarizer = _get_markdown_summarizer(markdown_summarizer) if enable_markdown_summaries else None
    embedding_helper = _get_embedding_client(embedding_client) if enable_embeddings else None

    def _build_metadata_record(
        *,
        source_file: Path,
        destination: Path,
        converter_name: str,
    ) -> dict[str, Any]:
        file_stat = source_file.stat()
        file_extension = source_file.suffix.lower() or None
        mime_type, _ = mimetypes.guess_type(source_file.name)
        file_type = mime_type or "application/octet-stream"
        metadata = {
            "source_path": str(source_file),
            "markdown_path": str(destination),
            "converter": converter_name,
            "size_bytes": file_stat.st_size,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "modified_at": datetime.fromtimestamp(file_stat.st_mtime, timezone.utc).isoformat(),
            "file_name": source_file.name,
            "file_extension": file_extension,
            "file_type": file_type,
            "description": "",
            "tags": [],
        }
        return metadata

    def _enrich_metadata(
        metadata: dict[str, Any],
        markdown_text: str,
        *,
        summarizer: MarkdownSummarizer | None,
        embedding_client: EmbeddingGemmaClient | None,
        source_file: Path,
    ) -> list[dict[str, Any]]:
        embeddings: list[dict[str, Any]] = []
        if summarizer is not None:
            try:
                summary = summarizer.summarize(markdown_text)
            except Exception as exc:  # pragma: no cover - best effort integration.
                logger.warning("Unable to summarize %s: %s", source_file, exc)
            else:
                metadata["description"] = summary.description
                metadata["tags"] = summary.tags
        if embedding_client is not None:
            try:
                document_embedding = embedding_client.embed(markdown_text)
                embeddings.append(
                    {
                        "source_path": metadata["source_path"],
                        "markdown_path": metadata["markdown_path"],
                        "variant": "document",
                        "vector": document_embedding,
                    },
                )
                if metadata["tags"]:
                    tag_text = "\n".join(metadata["tags"])
                    tag_embedding = embedding_client.embed(tag_text)
                    embeddings.append(
                        {
                            "source_path": metadata["source_path"],
                            "markdown_path": metadata["markdown_path"],
                            "variant": "tags",
                            "vector": tag_embedding,
                        },
                    )
            except Exception as exc:  # pragma: no cover - best effort integration.
                logger.warning("Unable to embed %s: %s", source_file, exc)
        return embeddings

    pending: list[tuple[Path, Path]] = []
    skipped = 0
    for source_file in files_to_index:
        relative_path = source_file.relative_to(base_path)
        destination = (target_root / relative_path).with_name(relative_path.name + ".md")
        if destination.exists():
            metadata_exists = metadata_store.has_record(source_file)
            doc_embedding_exists = (
                embedding_helper is None or embedding_store.has_variant(source_file, "document")
            )
            if not metadata_exists or not doc_embedding_exists:
                logger.info("Backfilling Lance artifacts for %s", relative_path)
                try:
                    markdown_text = destination.read_text(encoding="utf-8")
                except OSError as exc:  # pragma: no cover - best effort.
                    logger.warning("Unable to read existing markdown for %s: %s", relative_path, exc)
                else:
                    metadata = _build_metadata_record(
                        source_file=source_file,
                        destination=destination,
                        converter_name="unknown",
                    )
                    embeddings = _enrich_metadata(
                        metadata,
                        markdown_text,
                        summarizer=summarizer,
                        embedding_client=embedding_helper,
                        source_file=source_file,
                    )
                    metadata_store.upsert(metadata)
                    if embeddings:
                        embedding_store.upsert_many(embeddings)
            skipped += 1
            logger.info("Skipping %s (already indexed)", relative_path)
            continue
        pending.append((source_file, destination))

    if skipped:
        logger.info("Skipped %d previously indexed files in %s", skipped, base_path)

    iterable: Iterable[tuple[Path, Path]]
    if show_progress and pending:
        iterable = tqdm(
            pending,
            desc=f"Indexing {base_path.name}",
            unit="file",
            leave=False,
        )
    else:
        iterable = pending

    written_files: list[Path] = []
    for source_file, destination in iterable:
        markdown_text, converter_name = _convert_to_markdown(
            source_file,
            markitdown_converter=markitdown_converter,
            docling_converter=docling_converter,
            router=router,
        )
        logger.info("Converted %s using %s", source_file, converter_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown_text, encoding="utf-8")
        metadata = _build_metadata_record(
            source_file=source_file,
            destination=destination,
            converter_name=converter_name,
        )
        embeddings = _enrich_metadata(
            metadata,
            markdown_text,
            summarizer=summarizer,
            embedding_client=embedding_helper,
            source_file=source_file,
        )
        metadata_store.upsert(metadata)
        if embeddings:
            embedding_store.upsert_many(embeddings)
        written_files.append(destination)

    written_files.sort()
    return written_files


__all__ = [
    "list_files",
    "DEFAULT_EXTENSIONS",
    "DEFAULT_MARKDOWN_ROOT",
    "DEFAULT_METADATA_ROOT",
    "DEFAULT_EMBEDDING_ROOT",
    "build_markdown_index",
]
