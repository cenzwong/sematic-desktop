"""Integration tests for build_markdown_index additions."""
from __future__ import annotations

import json

from sematic_desktop.indexer import build_markdown_index
from sematic_desktop.summarizer import MarkdownSummary


class DummyMarkItDown:
    def convert(self, _: str) -> object:  # pragma: no cover - exercised indirectly.
        class Result:
            text_content = "# Title\n\nBody"

        return Result()


class DummySummarizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def summarize(self, markdown_text: str) -> MarkdownSummary:
        self.calls.append(markdown_text)
        return MarkdownSummary(description="desc", tags=["tag"])


def test_build_markdown_index_writes_summary_metadata(tmp_path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    (source_dir / "note.txt").write_text("hello world", encoding="utf-8")

    summarizer = DummySummarizer()
    outputs = build_markdown_index(
        source_dir,
        output_root=tmp_path / "markdown",
        metadata_root=tmp_path / "metadata",
        allowed_extensions=["txt"],
        markitdown_converter=DummyMarkItDown(),
        docling_converter=None,
        show_progress=False,
        markdown_summarizer=summarizer,
    )

    assert len(outputs) == 1
    metadata_path = tmp_path / "metadata" / "docs" / "note.txt.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["description"] == "desc"
    assert metadata["tags"] == ["tag"]
    assert summarizer.calls  # summarizer was invoked
