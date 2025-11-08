"""CLI entry point for the sematic-desktop project."""
from sematic_desktop import build_markdown_index


def main() -> None:
    """Materialize Markdown exports for the demo folder."""
    try:
        outputs = build_markdown_index("./my_folder")
    except Exception as exc:  # pragma: no cover - CLI guardrail.
        print(f"Failed to build markdown index: {exc}")
        return

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()