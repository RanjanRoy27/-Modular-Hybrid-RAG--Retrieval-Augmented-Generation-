# Architecture Roadmap

This project has a working `rag-platform/` runtime and a modular V3 query pipeline being developed alongside it. The immediate goal is to keep the live app stable while moving proven V3 behavior into the platform deliberately.

## Current Live Runtime

The active FastAPI app is powered by `rag-platform/`:

- `rag-platform/api/main.py`: FastAPI app, static UI serving, auth, sessions, file management, ingestion routes, answer routes, streaming, and domain selection.
- `rag-platform/api/schemas.py`: request and response models.
- `rag-platform/ingest/`: document loading, domain preprocessing, semantic chunking, embeddings, and Qdrant writes.
- `rag-platform/agent/`: tool-calling RAG agent with semantic search and document locator tools.
- `rag-platform/core/`: shared model clients, Qdrant retriever, chunker, reranker, and prompt helpers.
- `rag-platform/domains/`: canonical live domain model for `accounting`, `legal`, and `bookkeeping`.

Root-level `api.py`, `ingest.py`, and `main.py` remain as backward-compatible entry points. In this runtime, `/rag/answer` uses the `rag-platform/` agent path, and `/rag/stream` uses the platform retriever, reranker, and prompt chains directly.

## V3 Modular Direction

`pipeline.py` and `modules/` represent the intended V3 query architecture. This path is in progress and should not be described as the default live API behavior until it is wired into `rag-platform/api/main.py`.

Module responsibilities:

- `modules/config.py`: central configuration values.
- `modules/ingestion/`: loaders, cleaning, chunking, embedding, and ingestion orchestration.
- `modules/retrieval/`: vector search, BM25 keyword search, and hybrid RRF merging.
- `modules/ranking/`: reranking logic.
- `modules/generation/`: context building, prompt building, and LLM calls.
- `modules/query/`: query normalization and HyDE-style expansion.
- `modules/output/`: answer formatting and grounding checks.
- `modules/evaluation/`: evaluation helpers and datasets.
- `modules/observability/`: structured query logging/tracing.
- `modules/domain/`: experimental V3 domain detection for `real_estate`, `healthcare`, and `generic`.

## Live Data Flow

```text
Documents
  -> rag-platform/ingest loaders
  -> live domain preprocessor
  -> semantic chunking
  -> embeddings
  -> Qdrant
  -> platform retriever
  -> cross-encoder reranker
  -> platform agent or streaming QA chain
  -> Gemini
  -> answer with citations/source metadata
```

## Target V3 Data Flow

```text
Question
  -> normalize
  -> detect domain
  -> optionally expand with HyDE
  -> vector search + BM25 search
  -> RRF hybrid merge
  -> rerank
  -> build context
  -> build prompt
  -> Gemini
  -> format response
  -> grounding check and optional guard
  -> structured trace log
```

## Roadmap

Recommended integration order:

1. Align shared configuration between `rag-platform/` and `modules/config.py`.
2. Reconcile the live domain registry (`accounting`, `legal`, `bookkeeping`) with the experimental V3 detector taxonomy before exposing V3 through the API.
3. Replace or wrap platform ingestion with `modules/ingestion/pipeline.py` once output metadata and Qdrant writes match the live path.
4. Wire V3 retrieval behind a controlled API path or feature flag before replacing `/rag/answer`.
5. Move answer formatting, citation shaping, and grounding warnings into the platform response path.
6. Add tests around V3 pipeline behavior before making it the default runtime.
7. Keep `rag-platform/api/main.py` as the HTTP/session/static-file layer after core behavior is modularized.
