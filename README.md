# Modular Hybrid RAG

A FastAPI-based Retrieval-Augmented Generation app for chatting with private documents using Google Gemini, Qdrant, domain-aware runtime configuration, cross-encoder reranking, optional API-key auth, and a lightweight browser UI.

This repository currently has two important layers:

- **Live runtime:** `rag-platform/` powers the working API, UI, ingestion flow, domain routing, semantic retrieval, reranking, chat sessions, and streaming responses. Root files such as `api.py`, `ingest.py`, and `main.py` are compatibility entry points.
- **V3 modular pipeline:** `pipeline.py` and `modules/` contain an in-progress refactor with HyDE query expansion, BM25 + vector hybrid retrieval, Reciprocal Rank Fusion, grounding checks, observability, and evaluation helpers. These modules are not yet the default `/rag/answer` runtime path.

## Current Live Features

- Document ingestion for PDF, DOCX, TXT, and spreadsheet sources.
- Domain-isolated RAG layers for accounting, legal, and bookkeeping workflows.
- Qdrant-backed semantic retrieval over private documents.
- Google Gemini generation through LangChain.
- Tool-calling RAG agent with semantic search and document lookup tools.
- Cross-encoder reranking before answer generation.
- FastAPI API with an included static chat UI.
- Standard and streaming answer endpoints.
- Optional Bearer-token protection through `RAG_API_KEY`.
- Persistent chat sessions and uploaded document management.
- Docker and Docker Compose support for local or cloud deployment.

## V3 Work In Progress

The modular V3 path is being built in public under `modules/` and orchestrated by `pipeline.py`. It is intended to become the cleaner long-term query pipeline after integration work is complete.

V3 currently includes components for:

- Query normalization and HyDE-style query expansion.
- Domain detection for `real_estate`, `healthcare`, and `generic` queries.
- Vector retrieval, BM25 keyword retrieval, and RRF-based hybrid merging.
- Reranking, context building, prompt building, and output formatting.
- Pure-Python grounding checks with optional LLM guard logic for sensitive domains.
- Structured query logging and evaluation datasets/helpers.

Until V3 is explicitly wired into the FastAPI routes, treat these modules as experimental/in-progress rather than the live demo behavior.

## Tech Stack

- Python, FastAPI, Uvicorn
- LangChain and Google Gemini
- Qdrant local vector store
- Sentence Transformers cross-encoder reranker
- Static HTML/CSS/JavaScript frontend
- Docker Compose

## Repository Layout

```text
.
|-- rag-platform/
|   |-- core/                  # Shared embedding, retrieval, chunking, reranking, and LLM helpers
|   |-- domains/               # Accounting, legal, and bookkeeping domain modules
|   |-- api/                   # FastAPI app and Pydantic schemas
|   |-- agent/                 # Tool-calling RAG agent
|   |-- ingest/                # Document ingestion entry points
|   `-- tests/                 # Package-level structural tests
|-- modules/                   # Existing modular V3 components by responsibility
|-- static/                    # Browser UI
|-- eval/                      # Evaluation scripts and datasets
|-- tests/                     # Existing manual/API smoke tests
|-- scripts/                   # Utility scripts
|-- docs/                      # Project, deployment, and workflow docs
|-- api.py                     # Backward-compatible API entry point
|-- ingest.py                  # Backward-compatible ingestion entry point
|-- main.py                    # CLI chat entry point
|-- pipeline.py                # V3 query pipeline orchestrator, not yet the default API path
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- .env.example
```

## Domain Routing

The live API supports runtime domain selection:

```bash
POST /rag/answer?domain=accounting
POST /rag/answer?domain=legal
POST /rag/answer?domain=bookkeeping
```

Each live domain owns its own `config.yaml` and `preprocessor.py` under `rag-platform/domains/`. Adding a live domain requires a new folder there; the shared `rag-platform/core/` package does not need domain-specific changes.

The V3 detector in `modules/domain/` uses a separate experimental taxonomy (`real_estate`, `healthcare`, `generic`) and is not the live API domain model yet.

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

## API Shape

The live FastAPI app exposes routes for:

- `GET /health`
- `GET /rag/files`
- `POST /rag/upload`
- `DELETE /rag/files/{filename}`
- `POST /rag/ingest`
- `GET /rag/sessions`
- `POST /rag/sessions`
- `PUT /rag/sessions/{session_id}`
- `PUT /rag/sessions/{session_id}/messages`
- `PUT /rag/sessions/active/{session_id}`
- `DELETE /rag/sessions/{session_id}`
- `POST /rag/answer`
- `POST /rag/stream`

`/rag/answer` currently uses the `rag-platform/` runtime agent path. The V3 `pipeline.py` path is not yet wired in as the default answer implementation.

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
- [Architecture roadmap](docs/architecture.md)
- [V3 narrative design doc](docs/RAG_V3_Story_Mode_Explanation.md)
- [Historical progress report](docs/progress_report.md)
- [Changelog](CHANGELOG.md)
