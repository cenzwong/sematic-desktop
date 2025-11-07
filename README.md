# Sematic Desktop

Utility CLI that converts arbitrary documents into markdown and enriches them for semantic search experiments.

## Usage
- `uv sync` to install dependencies.
- `uv run python main.py` to convert everything under `./my_folder`. Each new artifact emits a markdown copy in `.semantic_index/markdown/<folder>` and a Lance dataset stored at `.semantic_index/metadata/<folder>.lance` that holds per-file metadata.

## Ollama Integration
- After converting a file, the pipeline now asks `gemma3:4b-it-qat` (via the local Ollama runtime) to summarize the generated markdown. The resulting `description` and `tags` fields are stored inside the Lance rows.
- Failures while calling Ollama do not abort indexing; errors are logged and the Lance dataset row is left untouched.
- Custom tooling can pass `enable_markdown_summaries=False` or provide a different `MarkdownSummarizer` when calling `build_markdown_index` to disable or override the behavior.

## Embedding Support
- Each markdown document also flows through `embeddinggemma:latest`, and its vector is stored alongside the metadata row. These Lance datasets will power a future LanceDB similarity query path.
