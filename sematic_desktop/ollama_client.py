"""Thin wrapper utilities for interacting with the Ollama CLI."""
from __future__ import annotations

import subprocess
from typing import Sequence


class OllamaError(RuntimeError):
    """Raised when the Ollama CLI fails to generate a response."""


class OllamaClient:
    """Minimal subprocess-based client for issuing prompts to Ollama."""

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
        if not prompt.strip():
            raise ValueError("Prompt must contain text.")

        command = [self.binary, "run", model]
        if options:
            command.extend(list(options))

        process = subprocess.run(
            command,
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=False,
            timeout=self.timeout,
            env=self.env,
        )
        if process.returncode != 0:
            stderr = process.stderr.decode("utf-8", errors="ignore").strip()
            raise OllamaError(
                f"Ollama exited with status {process.returncode}: {stderr or 'no stderr'}",
            )
        return process.stdout.decode("utf-8", errors="ignore").strip()


__all__ = ["OllamaClient", "OllamaError"]
