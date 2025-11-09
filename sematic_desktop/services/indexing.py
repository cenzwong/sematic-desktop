"""Business logic for the Markdown indexing pipeline."""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence

from sematic_desktop.data import LanceEmbeddingStore, LanceMetadataStore
from sematic_desktop.foundation.conversion import (
    build_conversion_plan,
    convert_with_docling,
    convert_with_markitdown,
)
from sematic_desktop.middleware import (
    ConversionRouter,
    EmbeddingGemmaClient,
    MarkdownSummarizer,
    gather_file_signals,
)

try:  # pragma: no cover - tqdm is optional during tests.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable.

    def tqdm(iterable, **_: Any):  # type: ignore[override]
        return iterable


try:  # pragma: no cover - exercised via dependency injection in tests.
    from markitdown import MarkItDown as _MarkItDownClass
except ImportError:  # pragma: no cover - dependency is optional at test time.
    _MarkItDownClass = None

try:  # pragma: no cover - exercised via dependency injection in tests.
    from docling.document_converter import DocumentConverter as _DoclingConverterClass
except ImportError:  # pragma: no cover - dependency is optional at test time.
    _DoclingConverterClass = None

logger = logging.getLogger(__name__)

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

__all__ = [
    "DEFAULT_EXTENSIONS",
    "DEFAULT_MARKDOWN_ROOT",
    "DEFAULT_METADATA_ROOT",
    "EmbeddingPersistenceService",
    "IndexingPipeline",
    "IndexingTask",
    "MarkdownIndexService",
    "MetadataPersistenceService",
    "build_markdown_index",
    "list_files",
]


def _normalized_extensions(extensions: Sequence[str] | None) -> set[str]:
    if not extensions:
        return set()
    return {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
    }


def list_files(
    folder: Path | str, *, allowed_extensions: Iterable[str] | None = None
) -> list[Path]:
    """Return indexed files contained within ``folder``."""
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


@dataclass(slots=True)
class IndexingTask:
    """Simple data structure describing a file slated for indexing."""

    source_path: Path
    destination_path: Path


@dataclass(slots=True)
class ConvertedDocument:
    """A markdown document produced from an ``IndexingTask``."""

    task: IndexingTask
    markdown_text: str
    converter_name: str


@dataclass(slots=True)
class EnrichedDocument:
    """Converted markdown plus metadata + embeddings ready for persistence."""

    converted: ConvertedDocument
    metadata: dict[str, Any]
    embeddings: list[dict[str, Any]]


@dataclass(slots=True)
class IndexingContext:
    """Shared objects that pipeline stages rely on."""

    base_path: Path
    target_root: Path
    metadata_store: LanceMetadataStore
    embedding_store: LanceEmbeddingStore
    summarizer: MarkdownSummarizer | None
    embedding_client: EmbeddingGemmaClient | None
    router: ConversionRouter
    markitdown_converter: Any | None
    docling_converter: Any | None


class PipelineStage(Protocol):
    """Minimal interface implemented by each indexing stage."""

    def run(self, items: Iterable[Any], context: IndexingContext) -> Iterable[Any]: ...


class IndexingPipeline:
    """Composable pipeline that chains together small indexing stages."""

    def __init__(self, stages: Sequence[PipelineStage]) -> None:
        self.stages = list(stages)

    def run(self, items: Iterable[Any], context: IndexingContext) -> list[Any]:
        data: Iterable[Any] = items
        for stage in self.stages:
            data = stage.run(data, context)
        return list(data)


class ConversionStage:
    """Turn ``IndexingTask`` instances into ``ConvertedDocument`` objects."""

    def run(
        self, items: Iterable[IndexingTask], context: IndexingContext
    ) -> Iterable[ConvertedDocument]:
        for task in items:
            markdown_text, converter_name = convert_to_markdown(
                task.source_path,
                router=context.router,
                markitdown_converter=context.markitdown_converter,
                docling_converter=context.docling_converter,
            )
            yield ConvertedDocument(
                task=task, markdown_text=markdown_text, converter_name=converter_name
            )


class EnrichmentStage:
    """Attach metadata + embeddings to converted documents."""

    def run(
        self, items: Iterable[ConvertedDocument], context: IndexingContext
    ) -> Iterable[EnrichedDocument]:
        for converted in items:
            metadata = build_metadata_record(
                source_file=converted.task.source_path,
                destination=converted.task.destination_path,
                converter_name=converted.converter_name,
            )
            embeddings = enrich_document(
                metadata=metadata,
                markdown_text=converted.markdown_text,
                summarizer=context.summarizer,
                embedding_client=context.embedding_client,
                source_file=converted.task.source_path,
            )
            yield EnrichedDocument(
                converted=converted, metadata=metadata, embeddings=embeddings
            )


class MetadataPersistenceService:
    """Wrap Lance metadata writes so they can be swapped or reused."""

    def __init__(self, store: LanceMetadataStore) -> None:
        self.store = store

    def write(self, metadata: dict[str, Any]) -> None:
        self.store.upsert(metadata)


class EmbeddingPersistenceService:
    """Wrap Lance embedding writes to isolate storage concerns."""

    def __init__(self, store: LanceEmbeddingStore) -> None:
        self.store = store

    def write_many(self, embeddings: list[dict[str, Any]]) -> None:
        if embeddings:
            self.store.upsert_many(embeddings)


class PersistenceStage:
    """Write markdown artifacts + Lance records to disk."""

    def __init__(
        self,
        metadata_service: MetadataPersistenceService,
        embedding_service: EmbeddingPersistenceService,
    ) -> None:
        self.metadata_service = metadata_service
        self.embedding_service = embedding_service

    def run(
        self, items: Iterable[EnrichedDocument], _: IndexingContext
    ) -> Iterable[Path]:
        for document in items:
            task = document.converted.task
            task.destination_path.parent.mkdir(parents=True, exist_ok=True)
            task.destination_path.write_text(
                document.converted.markdown_text, encoding="utf-8"
            )
            self.metadata_service.write(document.metadata)
            self.embedding_service.write_many(document.embeddings)
            yield task.destination_path


class MarkdownIndexService:
    """Coordinates conversion, enrichment, and Lance persistence."""

    def __init__(
        self,
        *,
        metadata_store_factory: Callable[[Path], LanceMetadataStore] | None = None,
        embedding_store_factory: Callable[[Path], LanceEmbeddingStore] | None = None,
        router: ConversionRouter | None = None,
        summarizer_factory: Callable[[], MarkdownSummarizer] | None = None,
        embedding_client_factory: Callable[[], EmbeddingGemmaClient] | None = None,
    ) -> None:
        self.metadata_store_factory = metadata_store_factory or _default_metadata_store
        self.embedding_store_factory = (
            embedding_store_factory or _default_embedding_store
        )
        self.router = router or ConversionRouter()
        self._summarizer_factory = summarizer_factory
        self._embedding_factory = embedding_client_factory
        self._summarizer: MarkdownSummarizer | None = None
        self._embedding_client: EmbeddingGemmaClient | None = None
        self._markitdown_instance: Any | None = None
        self._docling_instance: Any | None = None

    def build_index(
        self,
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
        metadata_store = self.metadata_store_factory(metadata_folder)
        embedding_store = self.embedding_store_factory(metadata_folder)
        metadata_service = MetadataPersistenceService(metadata_store)
        embedding_service = EmbeddingPersistenceService(embedding_store)

        files_to_index = list_files(
            base_path,
            allowed_extensions=(
                allowed_extensions if allowed_extensions is not None else []
            ),
        )
        summarizer = (
            markdown_summarizer
            if markdown_summarizer is not None
            else (
                self._get_markdown_summarizer() if enable_markdown_summaries else None
            )
        )
        embedding_helper = (
            embedding_client
            if embedding_client is not None
            else (self._get_embedding_client() if enable_embeddings else None)
        )
        router = router or self.router
        converter_context = IndexingContext(
            base_path=base_path,
            target_root=target_root,
            metadata_store=metadata_store,
            embedding_store=embedding_store,
            summarizer=summarizer,
            embedding_client=embedding_helper,
            router=router,
            markitdown_converter=markitdown_converter
            or self._get_markitdown_converter(),
            docling_converter=docling_converter or self._get_docling_converter(),
        )

        tasks = self._prepare_tasks(
            base_path=base_path,
            files_to_index=files_to_index,
            target_root=target_root,
            metadata_service=metadata_service,
            embedding_service=embedding_service,
            summarizer=summarizer,
            embedding_helper=embedding_helper,
        )

        if not tasks:
            return []

        iterable: Iterable[IndexingTask]
        if show_progress:
            iterable = tqdm(
                tasks, desc=f"Indexing {base_path.name}", unit="file", leave=False
            )
        else:
            iterable = tasks

        pipeline = IndexingPipeline(
            [
                ConversionStage(),
                EnrichmentStage(),
                PersistenceStage(metadata_service, embedding_service),
            ],
        )
        written_files = pipeline.run(iterable, converter_context)
        written_files.sort()
        return written_files

    def _prepare_tasks(
        self,
        *,
        base_path: Path,
        files_to_index: list[Path],
        target_root: Path,
        metadata_service: MetadataPersistenceService,
        embedding_service: EmbeddingPersistenceService,
        summarizer: MarkdownSummarizer | None,
        embedding_helper: EmbeddingGemmaClient | None,
    ) -> list[IndexingTask]:
        tasks: list[IndexingTask] = []
        skipped = 0
        for source_file in files_to_index:
            relative_path = source_file.relative_to(base_path)
            destination = (target_root / relative_path).with_name(
                relative_path.name + ".md"
            )
            if destination.exists():
                if self._backfill_existing(
                    source_file=source_file,
                    destination=destination,
                    metadata_service=metadata_service,
                    embedding_service=embedding_service,
                    summarizer=summarizer,
                    embedding_helper=embedding_helper,
                ):
                    skipped += 1
                continue
            tasks.append(
                IndexingTask(source_path=source_file, destination_path=destination)
            )

        if skipped:
            logger.info("Skipped %d previously indexed files in %s", skipped, base_path)
        return tasks

    def _backfill_existing(
        self,
        *,
        source_file: Path,
        destination: Path,
        metadata_service: MetadataPersistenceService,
        embedding_service: EmbeddingPersistenceService,
        summarizer: MarkdownSummarizer | None,
        embedding_helper: EmbeddingGemmaClient | None,
    ) -> bool:
        metadata_exists = metadata_service.store.has_record(source_file)
        doc_embedding_exists = (
            embedding_helper is None
            or embedding_service.store.has_variant(source_file, "document")
        )
        if metadata_exists and doc_embedding_exists:
            logger.info("Skipping %s (already indexed)", source_file.name)
            return True

        logger.info("Backfilling Lance artifacts for %s", source_file.name)
        try:
            markdown_text = destination.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - best effort.
            logger.warning(
                "Unable to read existing markdown for %s: %s", source_file, exc
            )
            return True

        metadata = build_metadata_record(
            source_file=source_file,
            destination=destination,
            converter_name="unknown",
        )
        embeddings = enrich_document(
            metadata,
            markdown_text,
            summarizer=summarizer,
            embedding_client=embedding_helper,
            source_file=source_file,
        )
        metadata_service.write(metadata)
        embedding_service.write_many(embeddings)
        return True

    def _get_markitdown_converter(
        self, markitdown_converter: Any | None = None
    ) -> Any | None:
        if markitdown_converter is not None:
            return markitdown_converter
        if self._markitdown_instance is None and _MarkItDownClass is not None:
            self._markitdown_instance = _MarkItDownClass()
        return self._markitdown_instance

    def _get_docling_converter(
        self, docling_converter: Any | None = None
    ) -> Any | None:
        if docling_converter is not None:
            return docling_converter
        if self._docling_instance is None and _DoclingConverterClass is not None:
            self._docling_instance = _DoclingConverterClass()
        return self._docling_instance

    def _get_markdown_summarizer(self) -> MarkdownSummarizer | None:
        if self._summarizer is not None:
            return self._summarizer
        if self._summarizer_factory is None:
            try:
                self._summarizer = MarkdownSummarizer()
            except (
                Exception
            ) as exc:  # pragma: no cover - fallback when Ollama fails to init.
                logger.warning("Disabling markdown summaries: %s", exc)
                self._summarizer = None
        else:
            try:
                self._summarizer = self._summarizer_factory()
            except (
                Exception
            ) as exc:  # pragma: no cover - fallback when Ollama fails to init.
                logger.warning("Unable to create markdown summarizer: %s", exc)
                self._summarizer = None
        return self._summarizer

    def _get_embedding_client(self) -> EmbeddingGemmaClient | None:
        if self._embedding_client is not None:
            return self._embedding_client
        factory = self._embedding_factory or EmbeddingGemmaClient
        try:
            self._embedding_client = factory()
        except Exception as exc:  # pragma: no cover - best effort integration.
            logger.warning("Disabling embeddings: %s", exc)
            self._embedding_client = None
        return self._embedding_client


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
    """Convenience wrapper that instantiates ``MarkdownIndexService``."""

    service = MarkdownIndexService()
    return service.build_index(
        folder,
        output_root=output_root,
        metadata_root=metadata_root,
        allowed_extensions=allowed_extensions,
        markitdown_converter=markitdown_converter,
        docling_converter=docling_converter,
        router=router,
        show_progress=show_progress,
        markdown_summarizer=markdown_summarizer,
        enable_markdown_summaries=enable_markdown_summaries,
        embedding_client=embedding_client,
        enable_embeddings=enable_embeddings,
    )


def _default_metadata_store(folder: Path) -> LanceMetadataStore:
    return LanceMetadataStore(folder, "properties")


def _default_embedding_store(folder: Path) -> LanceEmbeddingStore:
    return LanceEmbeddingStore(
        folder, doc_table_name="emb_doc", tag_table_name="emb_tags"
    )


def convert_to_markdown(
    source_path: Path,
    *,
    router: ConversionRouter,
    markitdown_converter: Any | None,
    docling_converter: Any | None,
) -> tuple[str, str]:
    """Convert ``source_path`` into markdown using the best available converter."""

    errors: list[str] = []
    signals = gather_file_signals(
        source_path,
        historical_success=router.historical_success_for(source_path.suffix.lower()),
    )

    plan = build_conversion_plan(router.plan_order(signals))

    best_attempt: tuple[str, str, float] | None = None

    for converter_name in plan.ordered_converters:
        try:
            if converter_name == "markitdown":
                markdown = convert_with_markitdown(
                    source_path, override=markitdown_converter
                )
            else:
                markdown = convert_with_docling(source_path, override=docling_converter)
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

            continue

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

    errors.append("no markdown converter available")
    raise RuntimeError(
        f"Unable to convert {source_path} to markdown ({'; '.join(errors)})"
    )


def build_metadata_record(
    *,
    source_file: Path,
    destination: Path,
    converter_name: str,
) -> dict[str, Any]:
    """Create the Lance metadata payload for ``source_file``."""

    file_stat = source_file.stat()
    file_extension = source_file.suffix.lower() or None
    mime_type, _ = mimetypes.guess_type(source_file.name)
    file_type = mime_type or "application/octet-stream"
    return {
        "source_path": str(source_file),
        "markdown_path": str(destination),
        "converter": converter_name,
        "size_bytes": file_stat.st_size,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "modified_at": datetime.fromtimestamp(
            file_stat.st_mtime, timezone.utc
        ).isoformat(),
        "file_name": source_file.name,
        "file_extension": file_extension,
        "file_type": file_type,
        "description": "",
        "tags": [],
    }


def enrich_document(
    metadata: dict[str, Any],
    markdown_text: str,
    *,
    summarizer: MarkdownSummarizer | None,
    embedding_client: EmbeddingGemmaClient | None,
    source_file: Path,
) -> list[dict[str, Any]]:
    """Populate metadata with summaries/tags and return embedding records."""

    summary = summarize_markdown(
        markdown_text, summarizer=summarizer, source_file=source_file
    )
    if summary is not None:
        metadata["description"] = summary.description
        metadata["tags"] = summary.tags
    embeddings = generate_embedding_records(
        metadata=metadata,
        markdown_text=markdown_text,
        embedding_client=embedding_client,
        source_file=source_file,
    )
    return embeddings


def summarize_markdown(
    markdown_text: str,
    *,
    summarizer: MarkdownSummarizer | None,
    source_file: Path,
):
    """Summarize markdown via the provided summarizer (best effort)."""

    if summarizer is None:
        return None
    try:
        return summarizer.summarize(markdown_text)
    except Exception as exc:  # pragma: no cover - best effort integration.
        logger.warning("Unable to summarize %s: %s", source_file, exc)
        return None


def generate_embedding_records(
    *,
    metadata: dict[str, Any],
    markdown_text: str,
    embedding_client: EmbeddingGemmaClient | None,
    source_file: Path,
) -> list[dict[str, Any]]:
    """Return embedding rows for the document + tag variants."""

    if embedding_client is None:
        return []

    records: list[dict[str, Any]] = []
    try:
        document_embedding = embedding_client.embed(markdown_text)
        records.append(
            {
                "source_path": metadata["source_path"],
                "markdown_path": metadata["markdown_path"],
                "variant": "document",
                "variant_label": None,
                "vector": document_embedding,
            },
        )
        if metadata.get("tags"):
            for tag in metadata["tags"]:
                tag_text = str(tag).strip()
                if not tag_text:
                    continue
                tag_embedding = embedding_client.embed(tag_text)
                records.append(
                    {
                        "source_path": metadata["source_path"],
                        "markdown_path": metadata["markdown_path"],
                        "variant": "tags",
                        "variant_label": tag_text,
                        "vector": tag_embedding,
                    },
                )
    except Exception as exc:  # pragma: no cover - best effort integration.
        logger.warning("Unable to embed %s: %s", source_file, exc)
    return records
