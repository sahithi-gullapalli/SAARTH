# 🧭 SAARTH — RAG-powered Multi-Agent AI Assistant

SAARTH answers questions from your own documents using retrieval-augmented generation (RAG),
a team of cooperating agents, tool calling, grounding checks and built-in evaluation.

```text
             ┌──────────┐   ┌────────┐   ┌───────────┐   ┌────────┐   ┌────────┐
question ──▶ │ planner  │──▶│ tools  │──▶│ retriever │──▶│ writer │──▶│ critic │──▶ answer
             └──────────┘   └────────┘   └───────────┘   └────────┘   └────────┘
                                 │             ▲                           │
                                 └─(no KB)─────┼──────▶ writer             │
                                               └──── revise (wider k) ◀────┘
```

| Agent | Role |
|---|---|
| **Planner** | Decides whether knowledge retrieval is needed or tools alone suffice |
| **Tools** | Native function calling (OpenAI / Azure / Bedrock) or rule-based fallback: calculator, date/time, word count |
| **Retriever** | Hybrid search — 70% dense vector similarity + 30% keyword overlap |
| **Writer** | Drafts an answer strictly from context, citing sources as `[n]` |
| **Critic** | Scores grounding; loops back for a wider search if weak, flags unverified answers |

## Tech stack

| Area | Used |
|---|---|
| Programming | Python 3.10+, OOP, type hints, pytest |
| LLMs | OpenAI API, Azure OpenAI, AWS Bedrock (Converse API), plus an offline mode with no key needed |
| RAG | LangChain text splitters, LlamaIndex document import |
| Orchestration | LangGraph (default), CrewAI (optional crew) |
| Embeddings / vectors | Sentence Transformers, OpenAI embeddings, FAISS, Chroma |
| Backend | FastAPI, Pydantic v2, REST |
| Storage | PostgreSQL + pgvector (HNSW index) |
| UI | Streamlit |
| Deployment | Git, GitHub Actions, Docker, docker-compose, AWS / Azure / GCP |
| Evaluation & monitoring | Built-in metrics, RAGAS, DeepEval, LangSmith tracing |

## Quick start (no API keys needed)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,ui]"
pytest                                                   # 13 tests
saarth ask "How many days of paid annual leave do employees get?"
saarth eval                                              # metrics on data/eval_set.json
saarth serve                                             # API on http://localhost:8000/docs
streamlit run ui/streamlit_app.py                        # chat UI on http://localhost:8501
```

The default configuration uses the **offline** model, hashing embeddings and an in-memory store,
so everything runs instantly. Switch to real models in `.env`.

## Using real models and stores

```bash
cp .env.example .env
pip install -e ".[all]"
```

| Goal | `.env` settings |
|---|---|
| OpenAI | `SAARTH_LLM_PROVIDER=openai`, `OPENAI_API_KEY=...` |
| Azure OpenAI | `SAARTH_LLM_PROVIDER=azure`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `SAARTH_AZURE_OPENAI_DEPLOYMENT` |
| AWS Bedrock | `SAARTH_LLM_PROVIDER=bedrock`, AWS credentials, `SAARTH_BEDROCK_MODEL_ID` |
| Sentence Transformers | `SAARTH_EMBEDDING_PROVIDER=sentence-transformers` |
| FAISS / Chroma | `SAARTH_VECTOR_BACKEND=faiss` or `chroma` |
| pgvector | `SAARTH_VECTOR_BACKEND=pgvector`, `DATABASE_URL=postgresql://...` |
| LangSmith | `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=...` |

If you change the embedding model, clear the vector store and re-ingest, because vector sizes differ.

## REST API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, active model, backend, chunk count |
| POST | `/ask` | `{"question": "...", "top_k": 4}` → answer, citations, tool results, grounding score, agent trace |
| POST | `/ingest` | Multipart upload of `.txt`, `.md`, `.pdf` files |
| POST | `/evaluate` | `[{"question": "...", "ground_truth": "..."}]` → per-sample and average metrics |

## Evaluation

* **Built-in** (offline): faithfulness, answer relevancy, context precision, context recall — `saarth eval`.
* **RAGAS**: `saarth.evaluation.evaluate_with_ragas(agent, samples)`.
* **DeepEval**: `saarth.evaluation.evaluate_with_deepeval(agent, samples)`.
* **LangSmith**: set `LANGSMITH_TRACING=true` and every LangGraph agent step is traced.

## Docker

```bash
docker compose up --build
# API → http://localhost:8000/docs   UI → http://localhost:8501   Postgres+pgvector → localhost:5432
```

## Cloud deployment

The same image runs on every major cloud. Push it to a registry, then:

* **AWS**: ECR + App Runner or ECS Fargate; RDS PostgreSQL with the `vector` extension; Bedrock for the LLM.
* **Azure**: ACR + Azure Container Apps; Azure Database for PostgreSQL (enable `vector`); Azure OpenAI.
* **GCP**: Artifact Registry + Cloud Run; Cloud SQL for PostgreSQL with pgvector.

Pass settings as environment variables or secrets. The included GitHub Actions workflow lints, tests and builds the image on every push.

## Project layout

```text
saarth/
  agents/graph.py      LangGraph multi-agent pipeline
  agents/crew.py       Optional CrewAI crew
  llm/                 OpenAI, Azure OpenAI, Bedrock, offline providers
  embeddings.py        Sentence Transformers / OpenAI / hashing
  vectorstores/        Memory, FAISS, Chroma, pgvector
  ingest.py            Loading + chunking (LangChain / LlamaIndex)
  retriever.py         Knowledge base + hybrid search
  tools.py             Safe calculator, date/time, word count
  evaluation.py        Built-in metrics, RAGAS, DeepEval
  api/main.py          FastAPI service
  cli.py               Command-line interface
ui/streamlit_app.py    Chat UI
tests/                 pytest suite
data/                  Sample documents + evaluation set
```

## Using CrewAI

```python
from saarth.retriever import KnowledgeBase
from saarth.agents.crew import ask_with_crew

kb = KnowledgeBase.from_settings(); kb.ingest_directory()
print(ask_with_crew(kb, "What is the expense claim deadline?"))
```
