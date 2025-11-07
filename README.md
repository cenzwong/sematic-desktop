# Sematic Desktop

Utility CLI that converts arbitrary documents into markdown and enriches them for semantic search experiments.

## Usage
- `uv sync` to install dependencies.
- `uv run python main.py` to convert everything under `./my_folder`. Each new artifact emits a markdown copy in `.semantic_index/markdown/<folder>` and metadata JSON under `.semantic_index/metadata/<folder>`.

## Ollama Integration
- After converting a file, the pipeline now asks `gemma3:4b-it-qat` (via the local Ollama runtime) to summarize the generated markdown. The resulting `description` and `tags` fields are appended to the metadata JSON.
- Failures while calling Ollama do not abort indexing; errors are logged and the metadata file is left untouched.
- Custom tooling can pass `enable_markdown_summaries=False` or provide a different `MarkdownSummarizer` when calling `build_markdown_index` to disable or override the behavior.

## Embedding Scaffold
- The repository also includes a placeholder `EmbeddingGemmaClient` plus a `write_embeddings_to_lance` stub so future work can wire up `embeddinggemma:latest` outputs and persist them inside a Lance dataset.
