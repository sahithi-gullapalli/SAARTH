"""LangSmith tracing + structured logging."""

from __future__ import annotations

import logging
import os

from saarth.config import Settings


def setup_observability(settings: Settings) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if settings.langsmith_tracing and os.getenv("LANGSMITH_API_KEY"):
        # LangGraph picks these up automatically and streams every agent step to LangSmith.
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        logging.getLogger("saarth").info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
