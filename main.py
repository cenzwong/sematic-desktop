"""CLI entry point for the sematic-desktop project."""
from pathlib import Path

from sematic_desktop.presentation.index_cli import print_index_results, run_indexing_cli


def main() -> None:
    """Materialize Markdown exports for the demo folder."""
    folder = Path("./my_folder")
    try:
        outputs = run_indexing_cli(folder)
    except Exception as exc:  # pragma: no cover - CLI guardrail.
        print(f"Failed to build markdown index: {exc}")
        return

    print_index_results(outputs)


if __name__ == "__main__":
    main()
