"""Business logic for semantic search workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sematic_desktop.data import LanceEmbeddingStore, LanceMetadataStore
from sematic_desktop.middleware import EmbeddingGemmaClient
from sematic_desktop.middleware.ollama import OllamaClient

__all__ = ["ContextAnswerer", "SearchHit", "SemanticSearchEngine"]


@dataclass(slots=True)
class SearchHit:
    """Normalized view over Lance search results."""

    source_path: str
    markdown_path: str
    description: str
    tags: list[str]
    score: float
    variant: str
    matched_tag: str | None = None


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
        return self._search(
            query,
            variant="tags",
            top_k=top_k,
            boost_exact_tags=True,
            oversample_factor=5,
        )

    def answer_question(self, question: str, *, top_k: int = 3) -> dict[str, Any]:
        """Answer ``question`` by grounding it in the most relevant documents."""
        hits = self.search_context(question, top_k=top_k)
        if not hits:
            return {"answer": "No matching documents were found.", "hits": []}
        contexts = []
        for hit in hits[:top_k]:
            snippet = self._read_markdown_snippet(hit.markdown_path)
            contexts.append(
                {
                    "source_path": hit.source_path,
                    "content": snippet if snippet else hit.description,
                },
            )
        answer = self.answerer.answer(question, contexts)
        return {"answer": answer, "hits": hits}

    def _search(
        self,
        query: str,
        *,
        variant: str,
        top_k: int,
        boost_exact_tags: bool = False,
        oversample_factor: int = 1,
    ) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("Query must contain text.")
        vector = self.embedding_client.embed(query)
        limit = max(top_k, top_k * max(1, oversample_factor))
        rows = self.embedding_store.search(vector, variant=variant, limit=limit)
        source_paths = [row["source_path"] for row in rows]
        metadata_map = self.metadata_store.fetch_by_paths(source_paths)
        hits_by_source: dict[str, SearchHit] = {}
        normalized_query = query.lower()
        for row in rows:
            metadata = metadata_map.get(row["source_path"], {})
            distance = float(row.get("_distance", 1.0))
            similarity = max(-1.0, min(1.0, 1.0 - distance))
            if boost_exact_tags:
                tags = [
                    tag.lower()
                    for tag in metadata.get("tags", [])
                    if isinstance(tag, str)
                ]
                if normalized_query in tags:
                    similarity = 1.0
            hit = SearchHit(
                source_path=row["source_path"],
                markdown_path=row["markdown_path"],
                description=str(metadata.get("description", "")),
                tags=list(metadata.get("tags", [])),
                score=similarity,
                variant=row["variant"],
                matched_tag=row.get("variant_label"),
            )
            existing = hits_by_source.get(hit.source_path)
            if existing is None or hit.score > existing.score:
                hits_by_source[hit.source_path] = hit
        hits = sorted(
            hits_by_source.values(), key=lambda item: item.score, reverse=True
        )
        return hits[:top_k]

    @staticmethod
    def _read_markdown_snippet(markdown_path: str, *, max_chars: int = 2_500) -> str:
        path = Path(markdown_path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return ""
        return text[:max_chars]
