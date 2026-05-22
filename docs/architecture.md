# Architecture

This project currently has two layers of code:

- Root-level runtime files that power the active MVP.
- `modules/` components that represent the cleaner V3 modular direction.

The long-term goal is to move behavior from root-level files into the module folders without breaking the working API.

## Runtime Entry Points

- `api.py`: FastAPI app, static UI serving, auth, sessions, ingestion routes, answer routes, and streaming.
- `ingest.py`: document loading, chunking, embedding, and Qdrant writes.
- `main.py`: CLI chat loop.
- `agent.py`: tool-calling agent and search/document tools.
- `core.py`: environment validation, model clients, Qdrant client, retriever, and prompt templates.

## Module Responsibilities

- `modules/config.py`: central configuration values.
- `modules/ingestion/`: loaders, cleaning, chunking, embedding, ingestion pipeline.
- `modules/retrieval/`: vector, BM25, and hybrid retrieval.
- `modules/ranking/`: reranking logic.
- `modules/generation/`: prompt building, context assembly, LLM calls.
- `modules/query/`: query normalization and expansion.
- `modules/output/`: answer formatting.
- `modules/evaluation/`: evaluation helpers and datasets.
- `modules/observability/`: logging/tracing.
- `modules/domain/`: domain detection.

## Data Flow

```text
Documents
  -> ingestion loader
  -> cleaner/chunker
  -> embeddings
  -> Qdrant
  -> retriever
  -> reranker
  -> context builder
  -> Gemini
  -> formatted answer with citations
```

## Refactor Direction

Future work should reduce logic in `api.py`, `core.py`, and `ingest.py` by moving stable behavior into `modules/`.

Recommended order:

1. Move configuration usage to `modules/config.py`.
2. Replace root ingestion logic with `modules/ingestion/pipeline.py`.
3. Replace root retrieval logic with `modules/retrieval/hybrid.py`.
4. Move answer formatting and citation shaping to `modules/output/formatter.py`.
5. Keep `api.py` as a thin HTTP layer.

