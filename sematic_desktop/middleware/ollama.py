"""Thin wrapper utilities for interacting with the Ollama CLI."""

from __future__ import annotations

from typing import Sequence

from sematic_desktop.foundation.ollama import run_ollama_prompt

__all__ = ["OllamaClient", "OllamaError"]


class OllamaError(RuntimeError):
    """Raised when the Ollama CLI fails to generate a response."""


class OllamaClient:
    """Minimal client for issuing prompts to Ollama."""

    def __init__(
        self,
        *,
        binary: str = "ollama",
        timeout: float = 120.0,
        env: dict[str, str] | None = None,
    ) -> None:
        self.binary = binary
        self.timeout = timeout
        self.env = env

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        options: Sequence[str] | None = None,
    ) -> str:
        """Send ``prompt`` to ``model`` and return the raw textual response."""
        try:
            return run_ollama_prompt(
                model,
                prompt,
                binary=self.binary,
                timeout=self.timeout,
                env=self.env,
                options=options,
            )
        except Exception as exc:  # pragma: no cover - best effort.
            raise OllamaError(str(exc)) from exc
