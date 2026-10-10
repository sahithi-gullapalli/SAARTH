"""Optional CrewAI orchestration (researcher + writer crew) over the same knowledge base.

Install with `pip install -e ".[crewai]"`; needs a hosted LLM key (e.g. OPENAI_API_KEY).
The LangGraph pipeline in `graph.py` is the default and also works fully offline.
"""

from __future__ import annotations

from typing import Any

from saarth.retriever import KnowledgeBase


def build_crew(kb: KnowledgeBase, question: str) -> Any:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool

    @tool("knowledge_search")
    def knowledge_search(query: str) -> str:
        """Search the SAARTH knowledge base and return numbered passages with sources."""
        hits = kb.search(query)
        return "\n\n".join(f"[{i}] ({h.source}) {h.text}" for i, h in enumerate(hits, 1)) or "No results."

    researcher = Agent(role="Researcher", goal="Find passages that answer the question",
                       backstory="Expert at searching internal knowledge bases.", tools=[knowledge_search])
    writer = Agent(role="Writer", goal="Write a concise answer citing passages as [n]",
                   backstory="Writes grounded answers and never invents facts.")
    research = Task(description=f"Research: {question}", expected_output="Relevant numbered passages",
                    agent=researcher)
    answer = Task(description=f"Answer using only the research: {question}",
                  expected_output="A grounded answer with [n] citations", agent=writer, context=[research])
    return Crew(agents=[researcher, writer], tasks=[research, answer], process=Process.sequential)


def ask_with_crew(kb: KnowledgeBase, question: str) -> str:
    return str(build_crew(kb, question).kickoff())
