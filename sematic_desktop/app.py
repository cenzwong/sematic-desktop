"""Primary surface for the sematic-desktop CLI."""
from __future__ import annotations


def get_greeting(target: str = "sematic-desktop") -> str:
    """Return the greeting emitted by the CLI."""
    return f"Hello from {target}!"


def run() -> None:
    """Execute the CLI greeting."""
    print(get_greeting())
