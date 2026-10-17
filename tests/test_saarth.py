from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from saarth.agents import SaarthAgent, grounding_score
from saarth.config import Settings, get_settings
from saarth.embeddings import HashingEmbedder
from saarth.evaluation import evaluate
from saarth.ingest import _builtin_split, chunk_document, split_text
from saarth.llm.base import BaseLLM, LLMResponse, Message, ToolCall
from saarth.retriever import KnowledgeBase
from saarth.schemas import EvalSample
from saarth.tools import calculator, current_datetime, detect_tool_calls, run_tool
from saarth.vectorstores import MemoryStore

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


@pytest.fixture()
def settings() -> Settings:
    return Settings(llm_provider="offline", embedding_provider="hashing", vector_backend="memory", docs_dir=str(DOCS))


@pytest.fixture()
def agent(settings: Settings) -> SaarthAgent:
    kb = KnowledgeBase.from_settings(settings)
    kb.ingest_directory()
    return SaarthAgent(kb, settings=settings)


def test_splitter_respects_size_and_overlap() -> None:
    text = " ".join(f"Sentence number {i} about retrieval." for i in range(200))
    for parts in (split_text(text, 200, 40), _builtin_split(text, 200, 40)):
        assert len(parts) > 5 and all(len(p) <= 200 for p in parts)
    with pytest.raises(ValueError):
        split_text(text, 100, 100)


def test_chunk_ids_are_stable() -> None:
    assert [c.id for c in chunk_document("a", "hello world")] == [c.id for c in chunk_document("a", "hello world")]


def test_embeddings_normalised_and_similar() -> None:
    v = HashingEmbedder(128).embed(["annual leave policy", "paid annual leave days", "docker container"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]


def test_memory_store_dedupes() -> None:
    store, e = MemoryStore(64), HashingEmbedder(64)
    chunks = chunk_document("x", "alpha beta gamma")
    store.add(chunks, e.embed([c.text for c in chunks]))
    store.add(chunks, e.embed([c.text for c in chunks]))
    assert store.count() == 1
    store.clear()
    assert store.search(e.embed_query("alpha"), 3) == []


def test_tools() -> None:
    assert calculator("(12+8)*3") == "(12+8)*3 = 60"
    assert "division by zero" in calculator("1/0")
    assert "Error" in calculator("__import__('os')")
    assert "UTC" in current_datetime("Not/AZone")
    assert "unknown tool" in run_tool("nope", {})
    assert detect_tool_calls("what is 15 * 4?") == [("calculator", {"expression": "15 * 4"})]
    assert detect_tool_calls("What time is it now in Asia/Kolkata?")[0] == ("current_datetime", {"timezone_name": "Asia/Kolkata"})


def test_retrieval_finds_right_doc(agent: SaarthAgent) -> None:
    assert agent.kb.search("How many days of paid annual leave do employees get?")[0].source == "company_policy.md"


def test_grounded_answer_with_citation(agent: SaarthAgent) -> None:
    r = agent.ask("How many days of paid annual leave do employees get?")
    assert "24 days" in r.answer
    assert r.citations and r.citations[0].source == "company_policy.md"
    assert r.grounded
    assert [s.agent for s in r.trace] == ["planner", "tools", "retriever", "writer", "critic"]


def test_tool_only_question_skips_retrieval(agent: SaarthAgent) -> None:
    r = agent.ask("Calculate 125 * 4")
    assert "500" in r.answer and r.tool_results[0]["tool"] == "calculator"
    assert "retriever" not in [s.agent for s in r.trace]


def test_unknown_question_refuses(agent: SaarthAgent) -> None:
    assert "enough information" in agent.ask("Zxqv plorb?").answer


class _HallucinatingLLM(BaseLLM):
    name, supports_tools = "fake", True

    def generate(self, messages: list[Message], tools: list | None = None) -> LLMResponse:
        if tools:
            return LLMResponse(tool_calls=[ToolCall(id="1", name="calculator", arguments={"expression": "2+2"})])
        return LLMResponse(content="Bananas fly purple rockets over Mars.")


def test_critic_revises_and_flags_ungrounded(settings: Settings) -> None:
    kb = KnowledgeBase.from_settings(settings)
    kb.ingest_directory()
    r = SaarthAgent(kb, llm=_HallucinatingLLM(), settings=settings).ask("Tell me about the leave policy")
    assert r.revisions == settings.max_revisions and not r.grounded
    assert "could not be fully verified" in r.answer
    assert r.tool_results[0]["output"] == "2+2 = 4"  # native tool calling path


def test_grounding_score() -> None:
    assert grounding_score("Leave is 24 days [1].", ["Employees get 24 days of leave."]) == 1.0
    assert grounding_score("Bananas fly purple rockets.", ["Employees get leave."]) == 0.0


def test_evaluation(agent: SaarthAgent) -> None:
    out = evaluate(agent, [EvalSample(question="What is the probation period?",
                                      ground_truth="The probation period is 6 months.")])
    assert out["summary"]["faithfulness"] >= 0.5 and out["summary"]["context_recall"] > 0.5


def test_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAARTH_DOCS_DIR", str(DOCS))
    monkeypatch.setenv("SAARTH_LLM_PROVIDER", "offline")
    monkeypatch.setenv("SAARTH_VECTOR_BACKEND", "memory")
    get_settings.cache_clear()
    from saarth.api.main import app

    with TestClient(app) as client:
        assert client.get("/health").json()["chunks"] > 0
        r = client.post("/ask", json={"question": "How often are performance reviews?"})
        assert r.status_code == 200 and "twice a year" in r.json()["answer"]
        assert client.post("/ask", json={"question": ""}).status_code == 422
        up = client.post("/ingest", files=[("files", ("n.txt", b"The office cafeteria opens at 8 am."))])
        assert up.json()["chunks"] == 1
        assert "8 am" in client.post("/ask", json={"question": "When does the cafeteria open?"}).json()["answer"]
        assert client.post("/ingest", files=[("files", ("a.exe", b"x"))]).status_code == 415
        ev = client.post("/evaluate", json=[{"question": "What is the probation period?",
                                              "ground_truth": "The probation period is 6 months."}])
        assert ev.status_code == 200 and "summary" in ev.json()
    get_settings.cache_clear()


def test_citations_only_reference_retrieved_chunks(agent: SaarthAgent) -> None:
    r = agent.ask("How many days of paid annual leave do employees get?")
    available_sources = len(agent.kb.search(
        "How many days of paid annual leave do employees get?",
        agent.settings.top_k,
    ))
    cited_indices = [c.index for c in r.citations]

    assert all(1 <= index <= available_sources for index in cited_indices)
    assert all(c.source for c in r.citations)

def test_critic_retry_retrieves_again(settings: Settings) -> None:
    kb = KnowledgeBase.from_settings(settings)
    kb.ingest_directory()
    agent = SaarthAgent(kb, llm=_HallucinatingLLM(), settings=settings)

    result = agent.ask("Tell me about the leave policy")

    agents = [step.agent for step in result.trace]
    retriever_positions = [
        i for i, name in enumerate(agents) if name == "retriever"
    ]

    assert len(retriever_positions) >= 2
    assert result.revisions == settings.max_revisions
    assert agents.count("critic") >= 2


def test_retry_uses_wider_retrieval(settings: Settings) -> None:
    kb = KnowledgeBase.from_settings(settings)
    kb.ingest_directory()
    agent = SaarthAgent(kb, llm=_HallucinatingLLM(), settings=settings)

    result = agent.ask("Tell me about the leave policy")

    retrieval_steps = [
        step.detail for step in result.trace if step.agent == "retriever"
    ]

    assert len(retrieval_steps) >= 2
    assert "k=" in retrieval_steps[0]
    assert "k=" in retrieval_steps[1]

    first_k = int(retrieval_steps[0].split("k=")[1].rstrip(")"))
    second_k = int(retrieval_steps[1].split("k=")[1].rstrip(")"))

    assert second_k > first_k
