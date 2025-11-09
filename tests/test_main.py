"""Tests for the sematic-desktop indexing helpers."""

from __future__ import annotations

import pytest

from sematic_desktop import list_files


def test_list_files_returns_sorted_paths(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    readme = tmp_path / "readme.txt"
    notes = docs_dir / "notes.md"
    skipped = tmp_path / "image.png"

    readme.write_text("content")
    notes.write_text("notes")
    skipped.write_text("binary", encoding="utf-8")

    result = list_files(tmp_path)
    expected = sorted([readme, notes], key=lambda path: str(path))

    assert result == expected


def test_list_files_accepts_custom_extensions(tmp_path) -> None:
    json_file = tmp_path / "payload.json"
    json_file.write_text("{}")
    (tmp_path / "notes.md").write_text("notes")

    result = list_files(tmp_path, allowed_extensions=["json"])
    assert result == [json_file]


def test_list_files_raises_for_missing_folder(tmp_path) -> None:
    missing_dir = tmp_path / "missing"
    with pytest.raises(ValueError, match="does not exist"):
        list_files(missing_dir)
