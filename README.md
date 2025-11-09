# Sematic Desktop

Utility CLI that converts arbitrary documents into markdown and enriches them for semantic search experiments.

## Usage
- `uv sync` to install dependencies.
- `uv run python main.py` to convert everything under `./my_folder`. Each run produces:
  - `.semantic_index/markdown/<folder>` — Markdown intermediates for every source file.
- `.semantic_index/metadata/<folder>/properties.lance` — Lance table holding structured metadata (paths, timestamps, tags, summaries).
- `.semantic_index/metadata/<folder>/emb_doc.lance` — Lance table containing per-document embeddings.
- `.semantic_index/metadata/<folder>/emb_tags.lance` — Lance table storing each tag embedding alongside the raw tag text for filtering/inspection.

## Ollama Integration
- After converting a file, the pipeline now asks `gemma3:4b-it-qat` (via the local Ollama runtime) to summarize the generated markdown. The resulting `description` and `tags` fields are stored inside the Lance rows.
- Failures while calling Ollama do not abort indexing; errors are logged and the Lance dataset row is left untouched.
- Custom tooling can pass `enable_markdown_summaries=False` or provide a different `MarkdownSummarizer` when calling `build_markdown_index` to disable or override the behavior.

## Embedding Support
- Each markdown document also flows through `embeddinggemma:latest`. The resulting vectors are tracked in the embeddings Lance dataset with two variants:
  - `document` — representation of the full markdown.
  - `tags` — representation of the auto-generated tags so tag searches remain semantic.
- These Lance datasets power higher-level APIs under `sematic_desktop.services.search`, enabling:
  - Context search — embed an arbitrary query and return the most similar markdown artifacts.
  - Tag search — embed tag-like queries and match against the tag vectors.
  - Ask with context — perform a context search, read the top markdown snippets, and feed them (plus the question) into `gemma3` to produce an answer that references the indexed documents.

## Semantic Search API
The `SemanticSearchEngine` bridges the Lance tables and Ollama so downstream consumers can run searches or question answering without reimplementing the plumbing.

```python
from sematic_desktop import (
    LanceEmbeddingStore,
    LanceMetadataStore,
    SemanticSearchEngine,
)

metadata_root = Path(".semantic_index/metadata/my_folder")
metadata_store = LanceMetadataStore(metadata_root, "properties")
embedding_store = LanceEmbeddingStore(metadata_root, doc_table_name="emb_doc", tag_table_name="emb_tags")
engine = SemanticSearchEngine(metadata_store, embedding_store)

print(engine.search_context("building amenities"))
print(engine.search_tags("sustainability"))
print(engine.answer_question("What are the lease terms?"))
```

All search modes rely on the embeddings produced during indexing, so re-run `uv run python main.py` anytime the source files or models change.

## Architecture
The repository now follows a layered layout to keep responsibilities isolated as the product grows toward a desktop GUI:
- `sematic_desktop/foundation` — low-level helpers that interact directly with Lance, Ollama, MarkItDown, and Docling.
- `sematic_desktop/data` — Lance persistence utilities (`LanceMetadataStore`, `LanceEmbeddingStore`) that provide the reusable data-access layer.
- `sematic_desktop/middleware` — focused helpers (embedding client, summarizer, routing) that compose the foundation primitives into higher-level behaviors.
- `sematic_desktop/services` — business logic for indexing and semantic search. The new `MarkdownIndexService` coordinates ingest + enrichment while `SemanticSearchEngine` handles query-time flows.
- `sematic_desktop/presentation` — CLI helpers powering `main.py` and `query_main.py`. Future GUI surfaces can reuse the same service layer without rewriting the plumbing.

Legacy compatibility shims were intentionally removed so new applications build directly on these layers without extra indirection.

### Function Philosophy
- Each function must perform a single task. If a helper starts combining multiple responsibilities (e.g., calling Ollama *and* writing to Lance), split it.
- Only foundation modules talk to external libraries; middleware/services import the foundation helpers instead of repeating integration code.
- Services/pipelines should be composed of small, testable stages. When adding a new feature, implement the primitive in `foundation/`, wrap it in middleware/data, then thread it through the relevant service.
