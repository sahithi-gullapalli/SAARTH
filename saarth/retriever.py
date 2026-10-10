"""Knowledge base: embeds, stores and retrieves chunks with hybrid (dense + keyword) re-ranking."""

from __future__ import annotations

from saarth.config import Settings, get_settings
from saarth.embeddings import Embedder, get_embedder
from saarth.ingest import chunk_document, load_directory
from saarth.schemas import Chunk, RetrievedChunk
from saarth.text import tokens
from saarth.vectorstores import VectorStore, get_vector_store


class KnowledgeBase:
    def __init__(self, embedder: Embedder, store: VectorStore, settings: Settings) -> None:
        self.embedder, self.store, self.settings = embedder, store, settings

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> KnowledgeBase:
        settings = settings or get_settings()
        embedder = get_embedder(settings)
        return cls(embedder, get_vector_store(embedder.dim, settings), settings)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        return len(chunks)

    def add_text(self, source: str, text: str) -> int:
        return self.add_chunks(chunk_document(source, text, self.settings.chunk_size, self.settings.chunk_overlap))

    def ingest_directory(self, directory: str | None = None) -> tuple[int, int]:
        docs = load_directory(directory or self.settings.docs_dir)
        return len(docs), sum(self.add_text(name, text) for name, text in docs)

    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.settings.top_k
        candidates = self.store.search(self.embedder.embed_query(query), k * 3)
        q = set(tokens(query))
        for c in candidates:
            keyword = len(q & set(tokens(c.text))) / max(len(q), 1)
            c.score = round(0.7 * c.score + 0.3 * keyword, 4)
        candidates.sort(key=lambda c: -c.score)
        return candidates[:k]
