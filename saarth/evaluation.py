
"""Evaluation: built-in RAG metrics plus adapters for RAGAS and DeepEval."""

from __future__ import annotations

from statistics import mean
from time import perf_counter
from typing import Any

from saarth.agents.graph import SaarthAgent, grounding_score
from saarth.schemas import EvalResult, EvalSample
from saarth.text import tokens


def _overlap(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    return len(ta & tb) / len(ta) if ta else 0.0


def evaluate_sample(agent: SaarthAgent, sample: EvalSample) -> EvalResult:
    start = perf_counter()
    resp = agent.ask(sample.question)
    latency_ms = round((perf_counter() - start) * 1000, 2)

    contexts = [c.text for c in agent.kb.search(sample.question)]
    relevant_contexts = [
        c for c in contexts
        if _overlap(sample.ground_truth, c) >= 0.3
    ]

    precision = (
        len(relevant_contexts) / len(contexts)
        if contexts else 0.0
    )

    # Token-overlap proxy, not a true ranked Recall@k metric yet.
    recall_at_k = _overlap(sample.ground_truth, " ".join(contexts))

    return EvalResult(
        question=sample.question,
        answer=resp.answer,
        faithfulness=grounding_score(
            resp.answer,
            contexts + [t["output"] for t in resp.tool_results],
        ),
        answer_relevancy=round(_overlap(sample.question, resp.answer), 3),
        context_precision=round(precision, 3),
        context_recall=round(recall_at_k, 3),
        latency_ms=latency_ms,
        recall_at_k=round(recall_at_k, 3),
    )


def evaluate(agent: SaarthAgent, samples: list[EvalSample]) -> dict[str, Any]:
    results = [evaluate_sample(agent, sample) for sample in samples]
    keys = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "recall_at_k",
        "latency_ms",
    )

    summary = (
        {key: round(mean(getattr(result, key) for result in results), 3)
         for key in keys}
        if results else {}
    )

    return {
        "summary": summary,
        "results": [result.model_dump() for result in results],
    }


def _collect(
    agent: SaarthAgent,
    samples: list[EvalSample],
) -> list[tuple[EvalSample, str, list[str]]]:
    return [
        (
            sample,
            agent.ask(sample.question).answer,
            [chunk.text for chunk in agent.kb.search(sample.question)],
        )
        for sample in samples
    ]


def evaluate_with_ragas(agent: SaarthAgent, samples: list[EvalSample]) -> Any:
    """Run RAGAS metrics; requires optional evaluation dependencies and judge configuration."""
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    rows = _collect(agent, samples)
    data = {
        "question": [sample.question for sample, _, _ in rows],
        "answer": [answer for _, answer, _ in rows],
        "contexts": [contexts for _, _, contexts in rows],
        "ground_truth": [sample.ground_truth for sample, _, _ in rows],
    }

    return ragas_evaluate(
        Dataset.from_dict(data),
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )


def evaluate_with_deepeval(
    agent: SaarthAgent,
    samples: list[EvalSample],
) -> Any:
    """Run DeepEval metrics; requires optional evaluation dependencies and judge configuration."""
    from deepeval import evaluate as de_evaluate
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    cases = [
        LLMTestCase(
            input=sample.question,
            actual_output=answer,
            expected_output=sample.ground_truth,
            retrieval_context=contexts,
        )
        for sample, answer, contexts in _collect(agent, samples)
    ]

    return de_evaluate(
        cases,
        [FaithfulnessMetric(), AnswerRelevancyMetric()],
    )
