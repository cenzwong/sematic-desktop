"""Routing heuristics for choosing the best document converter."""
from __future__ import annotations

from dataclasses import dataclass, field
import mimetypes
from pathlib import Path
import re
from typing import Any

try:  # pragma: no cover - optional dependency at runtime.
    import magic as _magic
except ImportError:  # pragma: no cover - fallback to mimetypes.
    _magic = None

__all__ = ["ConversionRouter", "FileSignals", "gather_file_signals"]

_DOCLING_FORWARD_SUFFIXES: set[str] = {
    ".pdf",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".doc",
    ".docx",
    ".odt",
    ".rtf",
    ".jpeg",
    ".jpg",
    ".png",
}

_MARKITDOWN_FIRST_SUFFIXES: set[str] = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
}


def _detect_mime_type(path: Path) -> str | None:
    if _magic is not None:
        try:
            return str(_magic.from_file(str(path), mime=True))
        except OSError:  # pragma: no cover - best effort only.
            pass
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed


@dataclass(slots=True)
class FileSignals:
    """Metadata collected from the source artifact to guide routing."""

    path: Path
    suffix: str
    size_bytes: int
    mime_type: str | None = None
    historical_success: dict[str, float] = field(default_factory=dict)

    @property
    def size_megabytes(self) -> float:
        return self.size_bytes / 1_000_000 if self.size_bytes else 0.0


def gather_file_signals(
    path: Path,
    *,
    historical_success: dict[str, float] | None = None,
) -> FileSignals:
    """Collects statistics that influence converter selection."""

    stat = path.stat()
    return FileSignals(
        path=path,
        suffix=path.suffix.lower(),
        size_bytes=stat.st_size,
        mime_type=_detect_mime_type(path),
        historical_success=dict(historical_success or {}),
    )


class ConversionRouter:
    """Scores available converters using routing heuristics and telemetry."""

    def __init__(
        self,
        *,
        large_file_threshold_mb: float = 8.0,
        expected_char_ratio: float = 0.15,
        history_weight: float = 0.5,
        historical_stats: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.large_file_threshold_mb = large_file_threshold_mb
        self.expected_char_ratio = expected_char_ratio
        self.history_weight = history_weight
        self._historical_stats: dict[str, dict[str, float]] = {
            suffix: dict(stats) for suffix, stats in (historical_stats or {}).items()
        }
        self.telemetry: list[dict[str, Any]] = []
        self._default_order = ["markitdown", "docling"]

    def historical_success_for(self, suffix: str) -> dict[str, float]:
        return dict(self._historical_stats.get(suffix, {}))

    def plan_order(self, signals: FileSignals) -> list[str]:
        """Return an ordered list of converters to try."""

        mark_score = 0.0
        doc_score = 0.0

        if signals.suffix in _DOCLING_FORWARD_SUFFIXES:
            doc_score += 2.0
        if signals.suffix in _MARKITDOWN_FIRST_SUFFIXES:
            mark_score += 1.5

        size_mb = signals.size_megabytes
        if size_mb >= self.large_file_threshold_mb:
            doc_score += 1.5
        elif size_mb <= 1.5:
            mark_score += 0.5

        mime = signals.mime_type or ""
        if mime.startswith("application/pdf"):
            doc_score += 2.5
        elif mime.startswith("text/"):
            mark_score += 1.0

        history = signals.historical_success
        if history:
            doc_score += history.get("docling", 0.0) * self.history_weight
            mark_score += history.get("markitdown", 0.0) * self.history_weight

        if doc_score > mark_score:
            return ["docling", "markitdown"]
        if mark_score > doc_score:
            return ["markitdown", "docling"]
        return list(self._default_order)

    def score_markdown(self, markdown: str, signals: FileSignals) -> float:
        """Approximate output quality combining length, structure, and density."""

        text = markdown.strip()
        if not text:
            return 0.0

        char_count = len(text)
        alpha_count = sum(1 for char in text if char.isalpha())
        alpha_ratio = alpha_count / char_count if char_count else 0.0

        expected_chars = max(20, int(signals.size_bytes * self.expected_char_ratio))
        length_score = min(1.0, char_count / max(expected_chars, 1))

        structural_bonus = 0.0
        if re.search(r"^#{1,6}\s", text, flags=re.MULTILINE):
            structural_bonus += 0.1
        if re.search(r"\|.+\|", text) and "---" in text:
            structural_bonus += 0.1

        unique_tokens = len(set(text.split()))
        diversity_score = min(1.0, unique_tokens / max(len(text.split()), 1))

        quality = (
            0.55 * length_score
            + 0.25 * alpha_ratio
            + 0.10 * diversity_score
            + structural_bonus
        )
        return float(max(0.0, min(1.0, quality)))

    def is_quality_acceptable(self, score: float, signals: FileSignals) -> bool:
        """Decide whether the markdown is good enough or if we should retry."""

        size = signals.size_bytes
        threshold = 0.3
        if size >= 8_000:
            threshold = 0.45
        if size >= int(self.large_file_threshold_mb * 1_000_000):
            threshold = 0.6
        return score >= threshold

    def record_outcome(
        self,
        signals: FileSignals,
        converter_name: str,
        *,
        success: bool,
        quality: float,
        error: str | None = None,
    ) -> None:
        """Track routing telemetry for future tuning."""

        self.telemetry.append(
            {
                "path": str(signals.path),
                "suffix": signals.suffix,
                "converter": converter_name,
                "success": success,
                "quality": quality,
                "error": error,
            }
        )

        suffix_stats = self._historical_stats.setdefault(signals.suffix, {})
        observed = 1.0 if success and error is None else 0.0
        previous = suffix_stats.get(converter_name, 0.5)
        suffix_stats[converter_name] = round((previous * 0.7) + (observed * 0.3), 3)

