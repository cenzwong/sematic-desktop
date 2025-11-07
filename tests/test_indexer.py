from pathlib import Path

import pytest

from sematic_desktop.indexer import DEFAULT_EXTENSIONS, list_files


def test_list_files_returns_all_nested_files(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested" / "child"
    nested_dir.mkdir(parents=True)
    files = [
        tmp_path / "root.txt",
        nested_dir / "deep.txt",
        tmp_path / "nested" / "sibling.md",
    ]
    for path in files:
        path.write_text(path.name)

    result = list_files(tmp_path)

    assert result == sorted(files)


@pytest.mark.parametrize("missing_path", ["missing", "file.txt"])
def test_list_files_rejects_invalid_directories(tmp_path: Path, missing_path: str) -> None:
    target = tmp_path / missing_path
    if not target.exists() and target.suffix == ".txt":
        target.write_text("not a directory")

    with pytest.raises(ValueError):
        list_files(target)


def test_list_files_filters_with_default_extensions(tmp_path: Path) -> None:
    allowed_file = tmp_path / "notes.txt"
    skipped_file = tmp_path / "payload.bin"
    allowed_file.write_text("ok")
    skipped_file.write_text("skip me")

    result = list_files(tmp_path)

    assert result == [allowed_file]
    assert ".txt" in DEFAULT_EXTENSIONS
    assert ".bin" not in DEFAULT_EXTENSIONS


def test_list_files_accepts_custom_extensions(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_text("data")

    result = list_files(tmp_path, allowed_extensions=["bin"])

    assert result == [payload]
