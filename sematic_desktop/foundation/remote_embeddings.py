"""HTTP helper for requesting embedding vectors from Ollama."""
from __future__ import annotations

import json
from typing import Any, Callable
from urllib import error, request

JsonBytes = bytes


def request_embedding_vector(
    payload: dict[str, Any],
    *,
    endpoint: str = "http://127.0.0.1:11434/api/embeddings",
    timeout: float = 120.0,
    transport: Callable[[JsonBytes], JsonBytes] | None = None,
) -> list[float]:
    """Send ``payload`` to Ollama's embedding endpoint and return the vector."""

    body = json.dumps(payload).encode("utf-8")
    raw = _send_request(body, endpoint=endpoint, timeout=timeout, transport=transport)
    data = json.loads(raw.decode("utf-8"))
    vector = _extract_embedding(data)
    if vector is None:
        raise RuntimeError("Embedding response did not include a vector.")
    return [float(value) for value in vector]


def _send_request(
    body: JsonBytes,
    *,
    endpoint: str,
    timeout: float,
    transport: Callable[[JsonBytes], JsonBytes] | None,
) -> JsonBytes:
    if transport is not None:
        return transport(body)

    req = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except error.URLError as exc:  # pragma: no cover - network failures are rare.
        raise RuntimeError(f"Failed to contact Ollama embeddings API: {exc}") from exc


def _extract_embedding(payload: dict[str, Any]) -> list[float] | None:
    candidate = payload.get("embedding")
    if isinstance(candidate, list):
        return candidate  # type: ignore[return-value]
    data = payload.get("data")
    if isinstance(data, list) and data:
        inner = data[0]
        if isinstance(inner, dict):
            vector = inner.get("embedding")
            if isinstance(vector, list):
                return vector  # type: ignore[return-value]
    return None

