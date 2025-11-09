"""Low-level helper for invoking the Ollama CLI."""

from __future__ import annotations

import subprocess
from typing import Sequence


def run_ollama_prompt(
    model: str,
    prompt: str,
    *,
    binary: str = "ollama",
    timeout: float = 120.0,
    env: dict[str, str] | None = None,
    options: Sequence[str] | None = None,
) -> str:
    """Send ``prompt`` to the Ollama CLI and return the response."""

    if not prompt.strip():
        raise ValueError("Prompt must contain text.")

    command = [binary, "run", model]
    if options:
        command.extend(list(options))

    process = subprocess.run(
        command,
        input=prompt.encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=timeout,
        env=env,
    )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(
            f"Ollama exited with status {process.returncode}: {stderr or 'no stderr'}"
        )
    return process.stdout.decode("utf-8", errors="ignore").strip()
