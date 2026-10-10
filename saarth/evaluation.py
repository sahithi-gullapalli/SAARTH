"""Evaluation: built-in RAG metrics plus adapters for RAGAS and DeepEval."""

from __future__ import annotations

from statistics import mean
from typing import Any

from saarth.agents.graph import SaarthAgent, grounding_score
from saarth.schemas import EvalResult, EvalSample
from saarth.text import tokens


def _overlap(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    return len(ta & tb) / len(ta) if ta else 0.0


def evaluate_sample(agent: SaarthAgent, sample: EvalSample) -> EvalResult:
    resp = agent.ask(sample.question)
    contexts = [c.text for c in agent.kb.search(sample.question)]
    precision = mean(1.0 if _overlap(sample.ground_truth, c) >= 0.3 else 0.0 for c in contexts) if contexts else 0.0
    return EvalResult(
        question=sample.question, answer=resp.answer,
        faithfulness=grounding_score(resp.answer, contexts + [t["output"] for t in resp.tool_results]),
        answer_relevancy=round(_overlap(sample.question, resp.answer), 3),
        context_precision=round(precision, 3),
        context_recall=round(_overlap(sample.ground_truth, " ".join(contexts)), 3),
    )


def evaluate(agent: SaarthAgent, samples: list[EvalSample]) -> dict[str, Any]:
    results = [evaluate_sample(agent, s) for s in samples]
    keys = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    summary = {k: round(mean(getattr(r, k) for r in results), 3) for k in keys} if results else {}
    return {"summary": summary, "results": [r.model_dump() for r in results]}


def _collect(agent: SaarthAgent, samples: list[EvalSample]) -> list[tuple[EvalSample, str, list[str]]]:
    return [(s, agent.ask(s.question).answer, [c.text for c in agent.kb.search(s.question)]) for s in samples]


def evaluate_with_ragas(agent: SaarthAgent, samples: list[EvalSample]) -> Any:
    """RAGAS metrics (pip install -e ".[eval]"; needs a hosted LLM key for the judge)."""
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    rows = _collect(agent, samples)
    data = {"question": [s.question for s, _, _ in rows], "answer": [a for _, a, _ in rows],
            "contexts": [c for _, _, c in rows], "ground_truth": [s.ground_truth for s, _, _ in rows]}
    return ragas_evaluate(Dataset.from_dict(data),
                          metrics=[faithfulness, answer_relevancy, context_precision, context_recall])


def evaluate_with_deepeval(agent: SaarthAgent, samples: list[EvalSample]) -> Any:
    """DeepEval metrics (pip install -e ".[eval]"; needs a hosted LLM key for the judge)."""
    from deepeval import evaluate as de_evaluate
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    cases = [LLMTestCase(input=s.question, actual_output=a, expected_output=s.ground_truth, retrieval_context=c)
             for s, a, c in _collect(agent, samples)]
    return de_evaluate(cases, [FaithfulnessMetric(), AnswerRelevancyMetric()])
