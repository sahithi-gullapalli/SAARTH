"""Shared Pydantic data models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(Chunk):
    score: float


class Citation(BaseModel):
    index: int
    source: str
    snippet: str
    score: float


class AgentStep(BaseModel):
    agent: str
    detail: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    tool_results: list[dict[str, Any]]
    grounding_score: float
    grounded: bool
    revisions: int
    trace: list[AgentStep]


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class EvalSample(BaseModel):
    question: str
    ground_truth: str


class EvalResult(BaseModel):
    question: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
