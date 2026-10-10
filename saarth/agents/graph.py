"""Multi-agent orchestration with LangGraph.

Agents (graph nodes):
    planner   -> decides whether knowledge retrieval is needed
    tools     -> tool-calling agent (native function calling or rule-based fallback)
    retriever -> hybrid search over the knowledge base
    writer    -> drafts a grounded answer with [n] citations
    critic    -> scores grounding; loops back to the retriever with a wider search if weak
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from saarth.config import Settings
from saarth.llm import BaseLLM, Message, get_llm
from saarth.retriever import KnowledgeBase
from saarth.schemas import AgentStep, AskResponse, Citation, RetrievedChunk
from saarth.text import sentences, tokens
from saarth.tools import TOOL_SCHEMAS, detect_tool_calls, run_tool

WRITER_SYSTEM = (
    "You are SAARTH, a helpful general-purpose AI assistant. "
    "Answer the user's question clearly and accurately using your general knowledge "
    "and any relevant numbered context or tool results provided. "
    "When you use information from numbered context, cite the relevant source "
    "using its number, like [1]. Do not invent citations. "
    "If no relevant context is available, answer normally without citations. "
    "If the question requires information you do not know, say so honestly. "
    "Be concise but helpful."
)
TOOL_SYSTEM = "Decide whether any tool is needed to answer the user. Call tools only when useful."
REFUSAL = "don't have enough information"
_TOOL_WORDS = {"calculate", "compute", "time", "date", "current", "today", "now", "utc", "plus", "times"}


class AgentState(TypedDict, total=False):
    question: str
    top_k: int
    needs_retrieval: bool
    contexts: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    answer: str
    grounding_score: float
    revise: bool
    revisions: int
    trace: list[dict[str, str]]



def grounding_score(answer: str, evidence: list[str]) -> float:
    """Estimate answer support from retrieved evidence."""
    if REFUSAL in answer.lower():
        return 1.0

    # General answers without retrieved evidence are not document-grounded.
    if not evidence:
        return 0.0

    vocab = set(tokens(" ".join(evidence)))
    sents = [re.sub(r"\[\d+\]", "", s) for s in sentences(answer)]
    sents = [s for s in sents if tokens(s)]

    if not sents or not vocab:
        return 0.0

    supported = sum(
        1 for s in sents
        if len(set(tokens(s)) & vocab) / max(1, len(set(tokens(s)))) >= 0.6
    )
    return round(supported / len(sents), 3)



class SaarthAgent:
    def __init__(self, kb: KnowledgeBase, llm: BaseLLM | None = None, settings: Settings | None = None) -> None:
        self.kb = kb
        self.settings = settings or kb.settings
        self.llm = llm or get_llm(self.settings)
        self.graph = self._build()

    @staticmethod
    def _log(state: AgentState, agent: str, detail: str) -> list[dict[str, str]]:
        return [*state.get("trace", []), {"agent": agent, "detail": detail}]

    # ---------- nodes ----------
    def planner(self, state: AgentState) -> AgentState:
        q = state["question"]
        detected = detect_tool_calls(q)
        leftover = q
        for _, args in detected:
            leftover = leftover.replace(str(args.get("expression", "")), " ")
        content = [t for t in tokens(leftover) if t not in _TOOL_WORDS]
        needs = not (detected and len(content) < 2)
        return {"needs_retrieval": needs, "revisions": 0, "revise": False, "tool_results": [], "contexts": [],
                "trace": self._log(state, "planner", f"retrieval={'yes' if needs else 'no'}")}

    def tools(self, state: AgentState) -> AgentState:
        q = state["question"]
        if self.llm.supports_tools:
            resp = self.llm.generate([Message(role="system", content=TOOL_SYSTEM), Message(role="user", content=q)],
                                     tools=TOOL_SCHEMAS)
            calls = [(c.name, c.arguments) for c in resp.tool_calls]
        else:
            calls = detect_tool_calls(q)
        results = [{"tool": n, "arguments": a, "output": run_tool(n, a)} for n, a in calls]
        detail = ", ".join(r["tool"] for r in results) or "no tools needed"
        return {"tool_results": results, "trace": self._log(state, "tools", detail)}

    def retriever(self, state: AgentState) -> AgentState:
        k = (state.get("top_k") or self.settings.top_k) * (1 + state.get("revisions", 0))
        hits = self.kb.search(state["question"], k)
        return {"contexts": [h.model_dump() for h in hits],
                "trace": self._log(state, "retriever", f"{len(hits)} chunks (k={k})")}

    def writer(self, state: AgentState) -> AgentState:
        ctx = "\n".join(f"[{i}] ({c['source']})\n{c['text']}" for i, c in enumerate(state.get("contexts", []), 1))
        tools_txt = "\n".join(r["output"] for r in state.get("tool_results", [])) or "None"
        prompt = f"Context:\n{ctx or 'None'}\nTool results:\n{tools_txt}\nQuestion: {state['question']}"
        answer = self.llm.generate([Message(role="system", content=WRITER_SYSTEM),
                                    Message(role="user", content=prompt)]).content.strip()
        return {"answer": answer, "trace": self._log(state, "writer", f"{len(answer)} chars drafted")}

    def critic(self, state: AgentState) -> AgentState:
        evidence = [c["text"] for c in state.get("contexts", [])] + [r["output"] for r in state.get("tool_results", [])]
        score = grounding_score(state.get("answer", ""), evidence)
        revise = bool(score < self.settings.grounding_threshold and state.get("needs_retrieval")
                      and state.get("revisions", 0) < self.settings.max_revisions)
        update: AgentState = {"grounding_score": score, "revise": revise,
                              "trace": self._log(state, "critic", f"grounding={score} -> {'revise' if revise else 'accept'}")}
        if revise:
            update["revisions"] = state.get("revisions", 0) + 1
        return update

    # ---------- graph ----------
    def _build(self) -> Any:
        g = StateGraph(AgentState)
        for name in ("planner", "tools", "retriever", "writer", "critic"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "planner")
        g.add_edge("planner", "tools")
        g.add_conditional_edges("tools", lambda s: "retriever" if s.get("needs_retrieval") else "writer",
                                {"retriever": "retriever", "writer": "writer"})
        g.add_edge("retriever", "writer")
        g.add_edge("writer", "critic")
        g.add_conditional_edges("critic", lambda s: "retriever" if s.get("revise") else "end",
                                {"retriever": "retriever", "end": END})
        return g.compile()

    def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        final: AgentState = self.graph.invoke({"question": question, "top_k": top_k or self.settings.top_k, "trace": []})
        contexts = [RetrievedChunk(**c) for c in final.get("contexts", [])]
        answer = final.get("answer", "")
        score = final.get("grounding_score", 0.0)
        grounded = score >= self.settings.grounding_threshold
        used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        if not grounded:
            answer += "\n\nNote: parts of this answer could not be fully verified against the knowledge base."
        citations = [Citation(index=i, source=c.source, snippet=c.text[:240], score=c.score)
                     for i, c in enumerate(contexts, 1) if i in used]
        return AskResponse(answer=answer, citations=citations, tool_results=final.get("tool_results", []),
                           grounding_score=score, grounded=grounded, revisions=final.get("revisions", 0),
                           trace=[AgentStep(**s) for s in final.get("trace", [])])
