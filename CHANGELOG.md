# Changelog

All notable project changes should be recorded here.

This project follows Semantic Versioning:

- `MAJOR` for incompatible API or data-contract changes.
- `MINOR` for new backwards-compatible features.
- `PATCH` for fixes, docs, and internal cleanup.

## [Unreleased]

### Added

- Modular V3 pipeline components under `modules/` for query normalization, HyDE-style expansion, vector + BM25 retrieval, RRF hybrid merging, reranking, grounding checks, observability, and evaluation.
- V3 query orchestrator in `pipeline.py`.

### Changed

- Documentation now distinguishes the live root-level FastAPI runtime from the in-progress V3 modular pipeline.
- Architecture notes reframed as a public roadmap.
- Historical FAISS-era progress report labeled as superseded.

## [0.1.1] - 2026-05-22

### Added

- Professional repository layout with `docs/`, `scripts/`, and `tests/`.
- Git workflow documentation for linear history and release tags.
- GitHub Actions syntax check workflow.

### Changed

- README rewritten as the project entry point.
- Documentation files moved under `docs/`.
- Utility scripts moved under `scripts/`.
- Manual smoke tests moved under `tests/`.

## [0.1.0] - 2026-05-22

### Added

- FastAPI RAG API and browser UI.
- Google Gemini model integration.
- Qdrant vector-store ingestion and retrieval.
- Optional API key authentication through `RAG_API_KEY`.
- Docker and Docker Compose deployment files.
