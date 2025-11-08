# Sematic Desktop

Utility CLI that converts arbitrary documents into markdown and enriches them for semantic search experiments.

## Usage
- `uv sync` to install dependencies.
- `uv run python main.py` to convert everything under `./my_folder`. Each run produces:
  - `.semantic_index/markdown/<folder>` — Markdown intermediates for every source file.
  - `.semantic_index/metadata/<folder>/properties.lance` — Lance table holding structured metadata (paths, timestamps, tags, summaries).
  - `.semantic_index/metadata/<folder>/embeddings.lance` — Lance vector store containing per-document and per-tag embeddings.

## Ollama Integration
- After converting a file, the pipeline now asks `gemma3:4b-it-qat` (via the local Ollama runtime) to summarize the generated markdown. The resulting `description` and `tags` fields are stored inside the Lance rows.
- Failures while calling Ollama do not abort indexing; errors are logged and the Lance dataset row is left untouched.
- Custom tooling can pass `enable_markdown_summaries=False` or provide a different `MarkdownSummarizer` when calling `build_markdown_index` to disable or override the behavior.

## Embedding Support
- Each markdown document also flows through `embeddinggemma:latest`. The resulting vectors are tracked in the embeddings Lance dataset with two variants:
  - `document` — representation of the full markdown.
  - `tags` — representation of the auto-generated tags so tag searches remain semantic.
- These Lance datasets power higher-level APIs under `sematic_desktop.search`, enabling:
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
embedding_store = LanceEmbeddingStore(metadata_root, "embeddings")
engine = SemanticSearchEngine(metadata_store, embedding_store)

print(engine.search_context("building amenities"))
print(engine.search_tags("sustainability"))
print(engine.answer_question("What are the lease terms?"))
```

All search modes rely on the embeddings produced during indexing, so re-run `uv run python main.py` anytime the source files or models change.
