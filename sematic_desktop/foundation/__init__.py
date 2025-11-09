"""Low-level helpers that interact with external libraries (Lance, Ollama, converters)."""

from .conversion import (
    ConversionPlan,
    build_conversion_plan,
    convert_with_docling,
    convert_with_markitdown,
    extract_markdown_from_docling,
    extract_markdown_from_markitdown,
)
from .lance import (
    LanceDocTable,
    LanceMetadataTable,
    LanceTagTable,
    create_doc_table,
    create_metadata_table,
    create_tag_table,
    delete_doc_vector,
    delete_tag_vector,
    fetch_metadata_rows,
    list_doc_sources,
    list_tag_pairs,
    search_vectors,
    upsert_metadata_row,
    upsert_vectors,
)
from .ollama import run_ollama_prompt
from .remote_embeddings import request_embedding_vector

__all__ = [
    "ConversionPlan",
    "LanceDocTable",
    "LanceMetadataTable",
    "LanceTagTable",
    "build_conversion_plan",
    "convert_with_docling",
    "convert_with_markitdown",
    "create_doc_table",
    "create_metadata_table",
    "create_tag_table",
    "delete_doc_vector",
    "delete_tag_vector",
    "extract_markdown_from_docling",
    "extract_markdown_from_markitdown",
    "fetch_metadata_rows",
    "list_doc_sources",
    "list_tag_pairs",
    "request_embedding_vector",
    "run_ollama_prompt",
    "search_vectors",
    "upsert_metadata_row",
    "upsert_vectors",
]
