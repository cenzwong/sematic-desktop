"""Tests for the sematic-desktop CLI helpers."""
from sematic_desktop import get_greeting, run


def test_get_greeting_returns_default_value() -> None:
    assert get_greeting() == "Hello from sematic-desktop!"


def test_run_prints_greeting(capsys) -> None:
    run()
    captured = capsys.readouterr()
    assert "Hello from sematic-desktop!" in captured.out
