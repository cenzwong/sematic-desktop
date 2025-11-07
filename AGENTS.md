# Repository Guidelines

## Project Goal
Deliver a desktop utility that walks any folder tree, indexes every file regardless of type, produces embeddings plus tags for each artifact, and surfaces them through a query-friendly UI. Keep ingest, embedding, storage, and UI layers decoupled so future agents can swap models or frontends without rewriting the pipeline.

## Project Structure & Module Organization
The repo is intentionally small: `main.py` hosts the CLI entry point, `pyproject.toml` defines project metadata, and `uv.lock` locks resolved dependencies. Create the package directory `sematic_desktop/` before adding modules so imports stay explicit. Place reusable assets such as sample data under `assets/` and keep documentation (including this file) at the top level. Mirror the runtime layout in tests (e.g., `tests/test_main.py`) to make ownership obvious.

## Build, Test, and Development Commands
- `uv sync` — install or update dependencies declared in `pyproject.toml` inside the managed virtual environment.
- `uv run python main.py` — run the desktop entry point locally; add feature flags via CLI args during development.
- `uv run pytest -q` — execute the entire test suite; pair with `-k name` for targeted runs.
- `uv run ruff check .` / `uv run ruff format` — lint and format the tree before committing to keep diffs minimal.

## Coding Style & Naming Conventions
Target Python 3.13 and idiomatic PEP 8 style: 4-space indentation, `snake_case` for functions and variables, `CapWords` for classes. Keep modules single-purpose and move helper logic from `main.py` into `sematic_desktop/<feature>.py`. Type annotate public functions, favor f-strings over concatenation, and include concise docstrings for side effects. Let Ruff (once added as a dev dependency) enforce unused-import and complexity limits.

## Testing Guidelines
Author Pytest cases under `tests/` with filenames mirroring the source (`tests/test_<module>.py`). Each test function should describe behavior, e.g., `test_main_prints_greeting`. When logic touches I/O, isolate it behind injectable helpers so tests can substitute fakes. Aim for ≥85% coverage on new code, use deterministic fixtures, and document any slow paths with markers such as `@pytest.mark.slow`.

## Commit & Pull Request Guidelines
History so far follows Conventional Commits (e.g., `feat: add tracing`); continue that format for easy changelog generation. Keep commits focused on a single concern and include brief bodies when context is not obvious. Pull requests must state motivation, link issues, and list verification steps (commands + status). Attach screenshots or recordings when behavior changes, request review before merging, and wait for CI to pass.
