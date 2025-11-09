"""Summarization helpers that lean on Ollama-powered Gemma models."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Sequence

from .ollama import OllamaClient

logger = logging.getLogger(__name__)

__all__ = ["MarkdownSummary", "MarkdownSummarizer"]


@dataclass(slots=True)
class MarkdownSummary:
    """Normalized structure returned by the markdown summarizer."""

    description: str
    tags: list[str]


class MarkdownSummarizer:
    """Use ``gemma3:4b-it-qat`` to summarize markdown and generate tags."""

    def __init__(
        self,
        *,
        client: OllamaClient | None = None,
        model: str = "gemma3:4b-it-qat",
        max_chars: int = 12_000,
    ) -> None:
        self.client = client or OllamaClient()
        self.model = model
        self.max_chars = max_chars

    def summarize(self, markdown_text: str) -> MarkdownSummary:
        """Return a concise description and tags for the document."""
        text = markdown_text.strip()
        if not text:
            raise ValueError("Markdown content is empty.")
        prompt = self._build_prompt(text)
        response = self.client.generate(self.model, prompt)
        payload = self._parse_response(response)
        description = (payload.get("description") or payload.get("summary") or "").strip()
        if not description:
            raise ValueError("Ollama response did not include a description.")
        tags = self._normalize_tags(payload.get("tags", []))
        return MarkdownSummary(description=description, tags=tags)

    def _build_prompt(self, markdown_text: str) -> str:
        content = markdown_text
        if len(content) > self.max_chars:
            content = content[: self.max_chars]
        instructions = (
            "You are an assistant that distills markdown documents. "
            "Read the content and respond with compact JSON that matches:\n"
            '{\n  "description": "2-3 sentence summary",\n'
            '  "tags": ["noun phrase 1", "noun phrase 2"]\n}\n'
            "Prefer 3-8 lower-case noun tags without punctuation. Do not explain the JSON."
        )
        return f"{instructions}\n\n<<<CONTENT START>>>\n{content}\n<<<CONTENT END>>>"

    def _parse_response(self, response: str) -> dict[str, Any]:
        raw = response.strip()
        for candidate in (raw, self._extract_json_block(raw)):
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        raise ValueError("Unable to parse JSON from Ollama response.")

    @staticmethod
    def _extract_json_block(text: str) -> str | None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if isinstance(tags, str):
            candidates: Sequence[str] = re.split(r"[;,]", tags)
        elif isinstance(tags, Sequence):
            candidates = tags
        else:
            return []

        normalized: list[str] = []
        for tag in candidates:
            if not isinstance(tag, str):
                continue
            cleaned = re.sub(r"[^a-z0-9\s\-]", "", tag.lower()).strip()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

