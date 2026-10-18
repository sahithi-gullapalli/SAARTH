# 🧭 SAARTH — Smart Autonomous Agentic Research & Task Hub

**A RAG-powered, multi-agent AI assistant for document-grounded question answering, tool execution, and intelligent research.**

SAARTH combines Retrieval-Augmented Generation (RAG), multi-agent orchestration, tool calling, grounding checks, and evaluation capabilities to help users interact with their documents through a conversational AI assistant.

## ✨ Key Features

- **Multi-agent orchestration:** Five cooperating agents coordinate planning, tool execution, retrieval, writing, and answer evaluation.
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document chunks to support answers with source citations.
- **Hybrid retrieval:** Combine dense-vector similarity and keyword overlap to find relevant content.
- **Tool integration:** Support calculations, date/time queries, and word-count operations through tool execution.
- **Grounding checks:** Evaluate whether answers are supported by available evidence.
- **Iterative retrieval:** Retry retrieval with a wider search when grounding is insufficient and retry limits allow it.
- **Multiple LLM providers:** Support configured OpenAI-compatible, Azure OpenAI, and AWS Bedrock integrations, along with offline testing.
- **REST API:** Expose assistant functionality through FastAPI.
- **Interactive UI:** Use a Streamlit interface to interact with SAARTH.
- **Evaluation and testing:** Run a built-in evaluation suite and automated tests.

## 🏗️ Architecture

SAARTH uses LangGraph to coordinate its five main agents.

```text
                         User Question
                               |
                               v
                         +-----------+
                         |  Planner  |
                         +-----+-----+
                               |
                               v
                         +-----------+
                         |   Tools   |
                         +-----+-----+
                               |
                    +----------+----------+
                    |                     |
              Retrieval needed?       No retrieval
                    |                     |
                    v                     |
              +-----------+               |
              | Retriever |               |
              +-----+-----+               |
                    |                     |
                    +----------+----------+
                               |
                               v
                         +-----------+
                         |  Writer   |
                         +-----+-----+
                               |
                               v
                         +-----------+
                         |  Critic   |
                         +-----+-----+
                               |
                     Grounding sufficient?
                          /          \
                        No            Yes
                        |              |
                        v              v
                 Retry retrieval     Final Answer
                 with wider search
```

The diagram represents the intended workflow. In the implemented graph, the planner routes through the tools agent, retrieval is conditional, and the critic can send the workflow back to the retriever when revision criteria are met.

### The five agents

| Agent | Responsibility |
|---|---|
| **Planner** | Determines whether document retrieval is needed for the question. |
| **Tools** | Executes useful tool calls through native function calling or a rule-based fallback. |
| **Retriever** | Searches the knowledge base for relevant document chunks. |
| **Writer** | Generates the response using the question, retrieved context, and tool results. |
| **Critic** | Evaluates grounding and requests another retrieval pass when appropriate. |

### Answer flow

1. The planner classifies the incoming question.
2. The tools agent executes relevant tools.
3. The retriever searches the knowledge base when retrieval is required.
4. The writer drafts the answer and can include numbered citations such as `[1]`.
5. The critic evaluates the answer against available evidence.
6. If grounding is insufficient and retries remain, retrieval is repeated with a wider search.
7. The final response includes available citations, tool results, grounding information, and agent trace details.

## 🛠️ Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python 3.10+, object-oriented programming, type hints |
| LLM providers | OpenAI API, Azure OpenAI, AWS Bedrock, offline provider |
| Orchestration | LangGraph; optional CrewAI integration |
| RAG and ingestion | LangChain text splitters, LlamaIndex document import |
| Embeddings | Sentence Transformers, OpenAI embeddings, hashing embeddings |
| Vector storage | In-memory backend, FAISS, Chroma, pgvector where configured |
| Backend | FastAPI, Pydantic, REST APIs |
| Database | PostgreSQL with pgvector where configured |
| Frontend | Streamlit |
| Testing and linting | pytest, Ruff |
| Deployment tooling | GitHub Actions, Docker, Docker Compose |
| Evaluation integrations | Built-in metrics, optional RAGAS and DeepEval integrations |

*Optional integrations require their respective dependencies, credentials, services, and configuration. Availability in the codebase does not by itself mean an integration has been tested in production.*

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or later
- Git
- A terminal or VS Code
- An API key only if you choose a hosted LLM provider

### 1. Clone the repository

```bash
git clone https://github.com/sahithi-gullapalli/SAARTH.git
cd SAARTH
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this in the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install the project

```bash
python -m pip install --upgrade pip
pip install -e ".[dev,ui]"
```

If the optional extras are unavailable in your current project configuration, check `pyproject.toml` for the supported installation extras.

### 4. Run the tests

```bash
pytest -q
```

The current local test suite has **16 passing tests**.

### 5. Run the linter

```bash
ruff check .
```

### 6. Try SAARTH in offline mode

The project supports an offline provider for local development and testing without a hosted LLM API key.

Set the following variables in Windows PowerShell:

```powershell
$env:SAARTH_LLM_PROVIDER = "offline"
$env:SAARTH_EMBEDDING_PROVIDER = "hashing"
$env:SAARTH_VECTOR_BACKEND = "memory"
```

Then try the CLI:

```bash
saarth ask "How many days of paid annual leave do employees get?"
```

The sample question depends on the sample documents available in your knowledge base. Offline mode does not require a hosted LLM API key, but its answers may be less capable than those from a hosted model.

### 7. Run the evaluation suite

```bash
saarth eval --file data/eval_set.json
```

The evaluation command uses the configured provider and knowledge base. For reproducible offline evaluation, keep the offline, hashing, and memory settings enabled in the same terminal.

### 8. Start the API

```bash
saarth serve
```

The API runs on port `8000` by default.

- API documentation: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/health

### 9. Start the Streamlit interface

Open another terminal, activate the same virtual environment, and run:

```bash
streamlit run ui/streamlit_app.py
```

The UI is typically available at http://localhost:8501.

Ensure the backend API is running if the UI is configured to communicate with it.

## 🔑 Configuring LLM Providers

To configure a hosted provider, create a local `.env` file using the project's example:

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

**Linux or macOS:**

```bash
cp .env.example .env
```

Edit `.env` with the settings supported by your current project configuration.

### Provider configuration examples

| Provider | Example settings |
|---|---|
| Offline | `SAARTH_LLM_PROVIDER=offline` |
| OpenAI | `SAARTH_LLM_PROVIDER=openai` and `OPENAI_API_KEY` |
| Azure OpenAI | `SAARTH_LLM_PROVIDER=azure`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, deployment setting |
| AWS Bedrock | `SAARTH_LLM_PROVIDER=bedrock`, AWS credentials, model ID |
| OpenAI-compatible service | Use the provider and base-URL settings supported by your configured LLM client |

For embedding configuration, the project supports hashing embeddings and configured alternatives such as Sentence Transformers or OpenAI embeddings.

For vector storage, select a backend supported by the installed project configuration.

**Security:** Never commit `.env`, API keys, access tokens, or other secrets to GitHub. Keep credentials in environment variables or an appropriate secrets manager.

### Changing embedding models

If you change embedding models, re-ingest your documents and rebuild the vector index as needed. Embedding dimensions and vector representations can differ between models.

## 📚 Document Ingestion and RAG

SAARTH uses a knowledge base to retrieve relevant chunks from ingested documents.

A typical RAG workflow is:

1. Load supported documents.
2. Split documents into smaller chunks.
3. Generate or calculate embeddings according to the selected provider.
4. Store document chunks in the configured vector backend.
5. Search for relevant content when a question requires retrieval.
6. Pass retrieved context to the writer.
7. Return the answer with citations when applicable.

The repository supports document ingestion workflows and exposes an API endpoint for file uploads.

Supported upload types documented by the API include:

- `.txt`
- `.md`
- `.pdf`

Actual parsing support depends on the installed ingestion dependencies and current configuration.

## 🌐 REST API

The FastAPI service exposes the following endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Reports service health and available runtime information. |
| `POST` | `/ask` | Sends a question and returns the assistant response. |
| `POST` | `/ingest` | Uploads supported documents for ingestion. |
| `POST` | `/evaluate` | Runs evaluation on supplied question and reference-answer samples. |

Use the interactive API documentation at http://localhost:8000/docs to inspect the actual request schemas and try the endpoints.

### Example question request

The `/ask` endpoint accepts a question and an optional retrieval limit, as supported by its request schema.

```json
{
  "question": "What is the expense claim deadline?",
  "top_k": 4
}
```

The response model can include:

- Answer text
- Citations
- Tool results
- Grounding score
- Grounded status
- Revision count
- Agent execution trace

The exact response fields and validation rules are defined by the current API schemas.

## 📊 Evaluation

SAARTH includes lightweight built-in metrics intended for development and regression testing.

| Metric | Purpose |
|---|---|
| Faithfulness / grounding | Estimates whether answer content is supported by available evidence. |
| Answer relevancy | Estimates answer alignment with the reference answer. |
| Context precision | Uses a heuristic to estimate how relevant retrieved chunks are. |
| Context recall | Uses a heuristic to estimate overlap with the expected answer. |
| Recall@k | The current implementation uses an overlap proxy; it is not a rigorous ranked-retrieval Recall@k measurement. |
| Latency | Measures end-to-end answer latency. |

**Important evaluation note:** The built-in metrics are heuristic proxies, not validated benchmark scores. A high score does not establish that every answer is correct or that every relevant document was retrieved.

The project also contains integration functions for RAGAS and DeepEval. Treat results from those tools as external evaluation results only after the relevant integrations have been configured and executed successfully.

### Run the evaluation suite

```bash
saarth eval --file data/eval_set.json
```

The evaluation dataset is maintained in `data/eval_set.json`. Baseline reports are available in:

- `data/baseline_offline.json`
- `data/baseline_offline_detailed.json`

These files record offline evaluation results from particular runs. Latency and other observed values can vary between runs and environments.

## 🐳 Docker

The repository includes Docker-related configuration.

If Docker and Docker Compose are installed and the configuration is complete, start the configured services with:

```bash
docker compose up --build
```

The expected service endpoints, if configured, are:

- API: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501
- PostgreSQL: localhost:5432

The actual services depend on the current Docker Compose configuration and environment variables. Review the compose file before assuming every optional service is enabled.

## ☁️ Cloud Deployment

SAARTH can be prepared for cloud deployment using container-based hosting, a configured LLM provider, and an appropriate vector or database backend.

Possible deployment paths include:

| Cloud | Potential deployment components |
|---|---|
| AWS | ECR, ECS Fargate or App Runner, RDS PostgreSQL, AWS Bedrock |
| Azure | Azure Container Registry, Azure Container Apps, Azure Database for PostgreSQL, Azure OpenAI |
| GCP | Artifact Registry, Cloud Run, Cloud SQL |

These are possible deployment architectures, not claims that a production deployment is already live. Each requires service configuration, credentials, networking, environment variables, and deployment validation.

## 🧪 Continuous Integration

The repository uses GitHub Actions to automate code-quality checks and tests. Docker-related checks may also be configured in the workflow.

The latest verified local checks were:

```text
Ruff: All checks passed
Pytest: 16 passed
```

The latest GitHub Actions run for commit `f19e851` also completed successfully.

View workflow history:

https://github.com/sahithi-gullapalli/SAARTH/actions

## 📁 Project Structure

```text
SAARTH/
├── saarth/
│   ├── agents/
│   │   ├── graph.py          # LangGraph agent orchestration
│   │   └── crew.py           # Optional CrewAI integration
│   ├── llm/                  # LLM providers
│   ├── vectorstores/         # Vector-store backends
│   ├── api/
│   │   └── main.py           # FastAPI application
│   ├── embeddings.py         # Embedding providers
│   ├── ingest.py             # Document ingestion and chunking
│   ├── retriever.py          # Knowledge base and retrieval
│   ├── tools.py              # Tool definitions and execution
│   ├── evaluation.py         # Evaluation utilities
│   ├── schemas.py            # Data models
│   └── cli.py                # Command-line interface
├── ui/
│   └── streamlit_app.py      # Streamlit interface
├── tests/
│   └── test_saarth.py        # Automated tests
├── data/
│   ├── eval_set.json         # Evaluation dataset
│   ├── baseline_offline.json
│   └── baseline_offline_detailed.json
├── .github/
│   └── workflows/            # CI configuration
├── .env.example              # Example environment configuration
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

Some optional modules and configuration files may differ between project versions. Refer to the repository for the authoritative file list.

## 🤝 Contributing

Contributions and suggestions are welcome.

A useful local validation workflow is:

```bash
ruff check .
pytest -q
```

Please avoid committing secrets, local virtual environments, generated caches, or personal configuration files.

## 🔒 Security

- Keep API keys and database credentials private.
- Use `.env` for local configuration and ensure it is excluded from Git.
- Use managed secrets for deployed environments.
- Validate uploaded files and API inputs.
- Apply appropriate authentication, authorization, request limits, and logging before exposing the service publicly.

## 👩‍💻 Author

**Sahithi Gullapalli**

Computer Science and Engineering graduate specializing in Cloud and Edge Computing, interested in multi-agent systems, LLM applications, and cloud-native development.

**Repository:** https://github.com/sahithi-gullapalli/SAARTH

---

*SAARTH — turning documents, tools, and cooperating agents into a more useful research assistant.*