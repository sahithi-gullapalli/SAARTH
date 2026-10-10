"""Embedding models: Sentence Transformers, OpenAI, and a dependency-free hashing embedder."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from saarth.config import Settings, get_settings
from saarth.text import tokens


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


class HashingEmbedder(Embedder):
    """Feature-hashed unigrams + bigrams. Fast, offline and deterministic."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _bucket(self, feat: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokens(text)
            for f in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]:
                idx, sign = self._bucket(f)
                out[i, idx] += sign
        return _normalise(out)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model: Any = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)


class OpenAIEmbedder(Embedder):
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self.client: Any = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.dim = 3072 if "large" in self.model else 1536

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 96):
            resp = self.client.embeddings.create(model=self.model, input=texts[start : start + 96])
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return _normalise(np.asarray(vectors, dtype=np.float32))


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    return HashingEmbedder(settings.embedding_dim)
