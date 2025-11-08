"""Search + Q&A helpers backed by Lance metadata and embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .embeddings import EmbeddingGemmaClient, LanceEmbeddingStore, LanceMetadataStore
from .ollama_client import OllamaClient


@dataclass(slots=True)
class SearchHit:
    """Normalized view over Lance search results."""

    source_path: str
    markdown_path: str
    description: str
    tags: list[str]
    score: float
    variant: str


class ContextAnswerer:
    """Turns ranked contexts into grounded answers via Ollama."""

    def __init__(
        self,
        *,
        client: OllamaClient | None = None,
        model: str = "gemma3:4b-it-qat",
        max_documents: int = 3,
        max_chars_per_doc: int = 2_000,
    ) -> None:
        self.client = client or OllamaClient()
        self.model = model
        self.max_documents = max_documents
        self.max_chars_per_doc = max_chars_per_doc

    def answer(self, question: str, contexts: Sequence[dict[str, str]]) -> str:
        """Return an answer that cites the supplied contexts."""
        question = question.strip()
        if not question:
            raise ValueError("Question must contain text.")
        if not contexts:
            raise ValueError("At least one context snippet is required.")
        prompt = self._build_prompt(question, contexts[: self.max_documents])
        return self.client.generate(self.model, prompt)

    def _build_prompt(self, question: str, contexts: Sequence[dict[str, str]]) -> str:
        blocks: list[str] = []
        for idx, context in enumerate(contexts, start=1):
            content = context.get("content", "")[: self.max_chars_per_doc]
            source = context.get("source_path", "unknown")
            blocks.append(f"Document {idx} (source: {source}):\n{content}")
        context_text = "\n\n".join(blocks)
        instructions = (
            "You are a helpful assistant with access to document snippets.\n"
            "Use ONLY the provided documents to answer the question.\n"
            "Cite the most relevant document when responding.\n"
        )
        return f"{instructions}\n{context_text}\n\nQuestion: {question}\nAnswer:"


class SemanticSearchEngine:
    """Search facade that supports context, tag, and QA workflows."""

    def __init__(
        self,
        metadata_store: LanceMetadataStore,
        embedding_store: LanceEmbeddingStore,
        *,
        embedding_client: EmbeddingGemmaClient | None = None,
        answerer: ContextAnswerer | None = None,
    ) -> None:
        self.metadata_store = metadata_store
        self.embedding_store = embedding_store
        self.embedding_client = embedding_client or EmbeddingGemmaClient()
        self.answerer = answerer or ContextAnswerer()

    def search_context(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        """Return documents ranked by markdown similarity."""
        return self._search(query, variant="document", top_k=top_k)

    def search_tags(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        """Return documents ranked by semantic tag similarity."""
        return self._search(query, variant="tags", top_k=top_k)

    def answer_question(self, question: str, *, top_k: int = 3) -> dict[str, Any]:
        """Answer ``question`` by grounding it in the most relevant documents."""
        hits = self.search_context(question, top_k=top_k)
        if not hits:
            return {"answer": "No matching documents were found.", "hits": []}
        contexts = []
        for hit in hits[: top_k]:
            snippet = self._read_markdown_snippet(hit.markdown_path)
            contexts.append(
                {
                    "source_path": hit.source_path,
                    "content": snippet if snippet else hit.description,
                },
            )
        answer = self.answerer.answer(question, contexts)
        return {"answer": answer, "hits": hits}

    def _search(self, query: str, *, variant: str, top_k: int) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("Query must contain text.")
        vector = self.embedding_client.embed(query)
        rows = self.embedding_store.search(vector, variant=variant, limit=top_k)
        source_paths = [row["source_path"] for row in rows]
        metadata_map = self.metadata_store.fetch_by_paths(source_paths)
        hits: list[SearchHit] = []
        for row in rows:
            metadata = metadata_map.get(row["source_path"], {})
            hits.append(
                SearchHit(
                    source_path=row["source_path"],
                    markdown_path=row["markdown_path"],
                    description=str(metadata.get("description", "")),
                    tags=list(metadata.get("tags", [])),
                    score=float(row.get("_distance", 0.0)),
                    variant=row["variant"],
                ),
            )
        return hits

    @staticmethod
    def _read_markdown_snippet(markdown_path: str, *, max_chars: int = 2_500) -> str:
        path = Path(markdown_path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return ""
        return text[:max_chars]


__all__ = ["ContextAnswerer", "SearchHit", "SemanticSearchEngine"]
