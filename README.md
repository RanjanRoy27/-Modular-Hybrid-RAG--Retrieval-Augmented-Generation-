# Modular Hybrid RAG

A FastAPI-based Retrieval-Augmented Generation system for chatting with private documents using Google Gemini, Qdrant, hybrid retrieval, reranking, and a lightweight browser UI.

## Features

- Document ingestion for PDF, DOCX, TXT, and spreadsheet sources.
- Domain-isolated RAG layers for accounting, legal, and bookkeeping workflows.
- Qdrant-backed semantic retrieval with optional BM25 keyword retrieval modules.
- Cross-encoder reranking before answer generation.
- FastAPI API with an included static chat UI.
- Optional Bearer-token protection through `RAG_API_KEY`.
- Persistent chat sessions and uploaded document management.
- Docker and Docker Compose support for local or cloud deployment.

## Tech Stack

- Python, FastAPI, Uvicorn
- LangChain and Google Gemini
- Qdrant local vector store
- Sentence Transformers reranker
- Static HTML/CSS/JavaScript frontend
- Docker Compose

## Repository Layout

```text
.
├── rag-platform/
│   ├── core/                  # Shared embedding, retrieval, chunking, reranking, and LLM helpers
│   ├── domains/               # Accounting, legal, and bookkeeping domain modules
│   ├── api/                   # FastAPI app and Pydantic schemas
│   ├── agent/                 # Tool-calling RAG agent
│   ├── ingest/                # Document ingestion entry points
│   └── tests/                 # Package-level structural tests
├── modules/                   # Existing modular V3 components by responsibility
├── static/                    # Browser UI
├── eval/                      # Evaluation scripts and datasets
├── tests/                     # Existing manual/API smoke tests
├── scripts/                   # Utility scripts
├── docs/                      # Project, deployment, and workflow docs
├── api.py                     # Backward-compatible API entry point
├── ingest.py                  # Backward-compatible ingestion entry point
├── main.py                    # CLI chat entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Domain Routing

The answer endpoint accepts a runtime `domain` query parameter:

```bash
POST /rag/answer?domain=accounting
POST /rag/answer?domain=legal
POST /rag/answer?domain=bookkeeping
```

Each domain owns its own `config.yaml` and `preprocessor.py`. Adding a domain requires a new folder under `rag-platform/domains/` with those files; the shared `rag-platform/core/` package does not need domain-specific changes.

## Quick Start

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure environment variables.

```bash
copy .env.example .env
```

Edit `.env` and set:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
RAG_API_KEY=your_private_app_api_key
```

`RAG_API_KEY` is optional for local development. When set, API clients must send:

```http
Authorization: Bearer your_private_app_api_key
```

4. Add documents to `data/`, then ingest them.

```bash
python ingest.py
```

5. Start the API and UI.

```bash
python api.py
```

You can also start the structured API module directly:

```bash
python rag-platform/api/main.py
```

Open `http://localhost:8000`.

## Docker

```bash
docker compose up --build
```

The Compose setup mounts:

- `./data` for source documents.
- `./qdrant_store` for vector storage.

## Testing

Run the API smoke test after the server is running:

```bash
$env:RAG_API_KEY="your_private_app_api_key"
python tests/test_api.py
```

Run a syntax-only check:

```bash
python -m py_compile api.py ingest.py main.py monitor.py pipeline.py rag-platform/api/main.py rag-platform/core/*.py rag-platform/agent/agent.py rag-platform/ingest/ingest.py
```

## Project Workflow

- Use small feature branches for new work.
- Keep `main` deployable.
- Prefer linear history with rebase or squash merges.
- Version releases with tags like `v0.1.0`, `v0.2.0`, and so on.

See [docs/git_workflow.md](docs/git_workflow.md) for the exact commit, branch, and release process.

## Documentation

- [Deployment guide](docs/deployment_guide.md)
- [Git workflow](docs/git_workflow.md)
- [Architecture notes](docs/architecture.md)
- [Changelog](CHANGELOG.md)
