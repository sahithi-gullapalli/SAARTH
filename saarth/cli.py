"""Command-line interface: `saarth ingest`, `saarth ask "..."`, `saarth eval`, `saarth serve`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="saarth")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="Ingest the docs directory").add_argument("--dir", default=None)
    sub.add_parser("ask", help="Ask a question").add_argument("question")
    sub.add_parser("eval", help="Run evaluation").add_argument("--file", default="data/eval_set.json")
    sub.add_parser("serve", help="Run the API").add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("saarth.api.main:app", host="0.0.0.0", port=args.port)
        return

    from saarth.agents import SaarthAgent
    from saarth.retriever import KnowledgeBase

    kb = KnowledgeBase.from_settings()
    if args.cmd == "ingest":
        docs, chunks = kb.ingest_directory(args.dir)
        print(f"Ingested {docs} documents into {chunks} chunks")
        return
    if kb.store.count() == 0:
        kb.ingest_directory()
    agent = SaarthAgent(kb)
    if args.cmd == "ask":
        r = agent.ask(args.question)
        print(r.answer)
        for c in r.citations:
            print(f"  [{c.index}] {c.source} (score {c.score:.2f})")
        print(f"grounding={r.grounding_score} revisions={r.revisions}")
    elif args.cmd == "eval":
        from saarth.evaluation import evaluate
        from saarth.schemas import EvalSample

        samples = [EvalSample(**s) for s in json.loads(Path(args.file).read_text(encoding="utf-8"))]
        print(json.dumps(evaluate(agent, samples)["summary"], indent=2))


if __name__ == "__main__":
    main()
