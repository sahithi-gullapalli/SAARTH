"""FastAPI REST service for SAARTH."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from saarth import __version__
from saarth.agents import SaarthAgent
from saarth.config import get_settings
from saarth.evaluation import evaluate
from saarth.observability import setup_observability
from saarth.retriever import KnowledgeBase
from saarth.schemas import AskRequest, AskResponse, EvalSample, IngestResponse

_state: dict[str, SaarthAgent] = {}


def get_agent() -> SaarthAgent:
    if "agent" not in _state:
        settings = get_settings()
        kb = KnowledgeBase.from_settings(settings)
        if kb.store.count() == 0:
            kb.ingest_directory()
        _state["agent"] = SaarthAgent(kb, settings=settings)
    return _state["agent"]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    setup_observability(get_settings())
    get_agent()
    yield
    _state.clear()


app = FastAPI(title="SAARTH API", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict[str, Any]:
    agent = get_agent()
    return {"status": "ok", "version": __version__, "llm": agent.llm.name,
            "vector_backend": agent.settings.vector_backend, "chunks": agent.kb.store.count()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(422, "Question must not be empty")
    try:
        return get_agent().ask(question, req.top_k)
    except Exception as exc:  # surface provider errors cleanly
        raise HTTPException(502, f"Assistant failed: {exc}") from exc


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    kb, total = get_agent().kb, 0
    for f in files:
        name = f.filename or "upload.txt"
        raw = await f.read()
        if name.lower().endswith(".pdf"):
            from pypdf import PdfReader

            text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
        elif name.lower().endswith((".txt", ".md")):
            text = raw.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(415, f"Unsupported file type: {name}")
        total += kb.add_text(name, text)
    return IngestResponse(documents=len(files), chunks=total)


@app.post("/evaluate")
def run_eval(samples: list[EvalSample]) -> dict[str, Any]:
    if not samples:
        raise HTTPException(422, "Provide at least one sample")
    return evaluate(get_agent(), samples)
