"""Central configuration loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["openai", "azure", "bedrock", "offline"]
EmbeddingProvider = Literal["sentence-transformers", "openai", "hashing"]
VectorBackend = Literal["memory", "faiss", "chroma", "pgvector"]


def _env(*names: str) -> AliasChoices:
    return AliasChoices(*names)


class Settings(BaseSettings):
    """Application settings. Fields can be overridden with SAARTH_<NAME> environment variables."""

    model_config = SettingsConfigDict(env_prefix="SAARTH_", env_file=".env", extra="ignore", populate_by_name=True)

    app_name: str = "SAARTH"

    # LLM
    llm_provider: LLMProvider = "offline"
    openai_api_key: str | None = Field(default=None, validation_alias=_env("OPENAI_API_KEY", "openai_api_key"))
    openai_base_url: str = "http://127.0.0.1:31415/v1"
    openai_model: str = "gpt-4o-mini"

    azure_openai_endpoint: str | None = Field(
        default=None, validation_alias=_env("AZURE_OPENAI_ENDPOINT", "azure_openai_endpoint"))
    azure_openai_api_key: str | None = Field(
        default=None, validation_alias=_env("AZURE_OPENAI_API_KEY", "azure_openai_api_key"))
    azure_openai_deployment: str = "gpt-4o-mini"
    azure_openai_api_version: str = "2024-06-01"
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    aws_region: str = Field(default="us-east-1", validation_alias=_env("AWS_REGION", "aws_region"))
    temperature: float = 0.1

    # Embeddings
    embedding_provider: EmbeddingProvider = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 384

    # Vector store
    vector_backend: VectorBackend = "memory"
    chroma_path: str = "./.chroma"
    faiss_path: str = "./.faiss"
    database_url: str = Field(default="postgresql://saarth:saarth@localhost:5432/saarth",
                              validation_alias=_env("DATABASE_URL", "database_url"))

    # Retrieval / agents
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    grounding_threshold: float = 0.55
    max_revisions: int = 2
    docs_dir: str = "./data/docs"

    # Monitoring
    langsmith_tracing: bool = Field(default=False, validation_alias=_env("LANGSMITH_TRACING", "langsmith_tracing"))
    langsmith_project: str = Field(default="saarth", validation_alias=_env("LANGSMITH_PROJECT", "langsmith_project"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
