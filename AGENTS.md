# Repository Guidelines

## Project Goal
Deliver a desktop utility that walks any folder tree, indexes every file regardless of type, produces embeddings plus tags for each artifact, and surfaces them through a query-friendly UI. Keep ingest, embedding, storage, and UI layers decoupled so future agents can swap models or frontends without rewriting the pipeline.

## Project Structure & Module Organization

The repository follows a strict three-layer layout so every component has a single responsibility:

1. **Foundation (`sematic_desktop/foundation/`)** — the only code that is allowed to touch third-party libraries. These helpers never contain business logic; they merely wrap APIs such as Ollama, Lance, MarkItDown, and Docling. Every function should do exactly one thing (e.g., “request an embedding vector” or “create a Lance table”) and return plain Python data structures.
2. **Middleware (`sematic_desktop/middleware/` and `sematic_desktop/data/`)** — these modules compose foundation helpers into reusable building blocks (embedding clients, summarizers, routing heuristics, Lance stores). Each function/class should orchestrate a single task (e.g., “generate embeddings for text” or “upsert metadata rows”) and expose a clean interface to the service layer.
3. **Services & Presentation (`sematic_desktop/services/` and `sematic_desktop/presentation/`)** — user-facing APIs and CLIs. Services wire middleware components into pipelines (indexing, search), while presentation modules provide CLI entry points. No service may call foundation helpers directly; everything must flow foundation → middleware/data → services → presentation.

Tests in `tests/` mirror this structure (foundation logic is probed indirectly via middleware/services).

### Artifact Layout
- Markdown intermediates live under `.semantic_index/markdown/<folder>`.
- Lance metadata lives under `.semantic_index/metadata/<folder>/properties.lance`.
- Lance embeddings are split into `.semantic_index/metadata/<folder>/emb_doc.lance` and `.semantic_index/metadata/<folder>/emb_tags.lance`.
Keeping ingest, enrichment, storage, and query components decoupled lets future agents swap any layer without rewriting the entire pipeline.

## Build, Test, and Development Commands
- `uv sync` — install or update dependencies declared in `pyproject.toml` inside the managed virtual environment.
- `uv run python main.py` — run the desktop entry point locally; add feature flags via CLI args during development.
- `uv run pytest -q` — execute the entire test suite; pair with `-k name` for targeted runs.
- `uv run ruff check .` / `uv run ruff format` — lint and format the tree before committing to keep diffs minimal.

## Coding Style & Naming Conventions

- Target Python 3.13 / PEP 8: 4-space indentation, `snake_case` functions, `CapWords` classes.
- **Single task per function.** If a function performs two conceptual operations, split it. Helpers should be composable and side effects should be explicit.
- Only foundation modules may import third-party libraries such as `lancedb`, `docling`, `markitdown`, or `subprocess` (for Ollama). Middleware and services must rely on the foundation helpers instead of reusing raw APIs.
- Services must compose middleware/data objects through dependency injection so tests can provide fakes.
- Add concise docstrings describing the side effect or return value; prefer type annotations for public APIs.
- Favor functional composition: pipeline stages (`ConversionStage`, `EnrichmentStage`, etc.) are single-purpose iterators that accept data and yield the transformed result.

## Testing Guidelines
Author Pytest cases under `tests/` with filenames mirroring the source (`tests/test_<module>.py`). Each test function should describe behavior, e.g., `test_main_prints_greeting`. When logic touches I/O, isolate it behind injectable helpers so tests can substitute fakes. Aim for ≥85% coverage on new code, use deterministic fixtures, and document any slow paths with markers such as `@pytest.mark.slow`.

- Tests that cover Lance-backed behavior should use temporary directories and the helper stores directly to avoid polluting the working tree.
- When adding new search behaviors, pair them with high-level tests under `tests/test_search.py` to ensure embeddings + metadata stay in sync.

## Commit & Pull Request Guidelines
History so far follows Conventional Commits (e.g., `feat: add tracing`); continue that format for easy changelog generation. Keep commits focused on a single concern and include brief bodies when context is not obvious. Pull requests must state motivation, link issues, and list verification steps (commands + status). Attach screenshots or recordings when behavior changes, request review before merging, and wait for CI to pass.
