"""Vector store backends: in-memory (NumPy), FAISS, Chroma and PostgreSQL/pgvector."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from saarth.config import Settings, get_settings
from saarth.schemas import Chunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...


class MemoryStore(VectorStore):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        existing = {c.id for c in self.chunks}
        keep = [i for i, c in enumerate(chunks) if c.id not in existing]
        if not keep:
            return
        self.chunks.extend(chunks[i] for i in keep)
        self.matrix = np.vstack([self.matrix, vectors[keep].astype(np.float32)])

    def search(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        scores = self.matrix @ vector.astype(np.float32)
        order = np.argsort(-scores)[:k]
        return [RetrievedChunk(**self.chunks[i].model_dump(), score=float(scores[i])) for i in order]

    def count(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks, self.matrix = [], np.zeros((0, self.dim), dtype=np.float32)


class FaissStore(VectorStore):
    """Inner-product FAISS index (vectors are normalised, so IP == cosine). Persists to disk."""

    def __init__(self, dim: int, path: str) -> None:
        import faiss

        self.faiss: Any = faiss
        self.dim, self.path = dim, path
        self.index_file, self.meta_file = os.path.join(path, "index.faiss"), os.path.join(path, "meta.json")
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, encoding="utf-8") as f:
                self.chunks = [Chunk(**c) for c in json.load(f)]
        else:
            self.index, self.chunks = faiss.IndexFlatIP(dim), []

    def _persist(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        self.faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in self.chunks], f)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        existing = {c.id for c in self.chunks}
        keep = [i for i, c in enumerate(chunks) if c.id not in existing]
        if not keep:
            return
        self.index.add(np.ascontiguousarray(vectors[keep], dtype=np.float32))
        self.chunks.extend(chunks[i] for i in keep)
        self._persist()

    def search(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        scores, ids = self.index.search(np.asarray([vector], dtype=np.float32), min(k, len(self.chunks)))
        return [RetrievedChunk(**self.chunks[i].model_dump(), score=float(s))
                for s, i in zip(scores[0], ids[0], strict=True) if i >= 0]

    def count(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.index, self.chunks = self.faiss.IndexFlatIP(self.dim), []
        self._persist()


class ChromaStore(VectorStore):
    def __init__(self, path: str, collection: str = "saarth") -> None:
        import chromadb

        self.client: Any = chromadb.PersistentClient(path=path)
        self.name = collection
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        self.col.upsert(
            ids=[c.id for c in chunks], embeddings=vectors.tolist(), documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, **{k: str(v) for k, v in c.metadata.items()}} for c in chunks],
        )

    def search(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        n = self.count()
        if n == 0:
            return []
        r = self.col.query(query_embeddings=[vector.tolist()], n_results=min(k, n))
        out = []
        for cid, doc, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0], strict=True):
            meta = dict(meta or {})
            source = str(meta.pop("source", "unknown"))
            out.append(RetrievedChunk(id=cid, text=doc, source=source, metadata=meta, score=1 - float(dist)))
        return out

    def count(self) -> int:
        return int(self.col.count())

    def clear(self) -> None:
        self.client.delete_collection(self.name)
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})


class PgVectorStore(VectorStore):
    def __init__(self, dsn: str, dim: int) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn: Any = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(
            f"""CREATE TABLE IF NOT EXISTS saarth_chunks (
                id TEXT PRIMARY KEY, text TEXT NOT NULL, source TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}', embedding vector({int(dim)}) NOT NULL)"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS saarth_chunks_hnsw ON saarth_chunks "
                          "USING hnsw (embedding vector_cosine_ops)")

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO saarth_chunks (id, text, source, metadata, embedding) VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, embedding=EXCLUDED.embedding",
                [(c.id, c.text, c.source, json.dumps(c.metadata), v) for c, v in zip(chunks, vectors, strict=True)],
            )

    def search(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        rows = self.conn.execute(
            "SELECT id, text, source, metadata, 1 - (embedding <=> %s) AS score "
            "FROM saarth_chunks ORDER BY embedding <=> %s LIMIT %s", (vector, vector, k)).fetchall()
        return [RetrievedChunk(id=r[0], text=r[1], source=r[2], metadata=r[3] or {}, score=float(r[4])) for r in rows]

    def count(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM saarth_chunks").fetchone()[0])

    def clear(self) -> None:
        self.conn.execute("TRUNCATE saarth_chunks")


def get_vector_store(dim: int, settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    if settings.vector_backend == "faiss":
        return FaissStore(dim, settings.faiss_path)
    if settings.vector_backend == "chroma":
        return ChromaStore(settings.chroma_path)
    if settings.vector_backend == "pgvector":
        return PgVectorStore(settings.database_url, dim)
    return MemoryStore(dim)
