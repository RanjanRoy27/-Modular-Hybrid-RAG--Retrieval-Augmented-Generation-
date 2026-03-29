"""
ingestion/pipeline.py — Ingestion Orchestrator
Responsibility: Coordinate loader → cleaner → chunker → embedder.

Dirty-flag wiring (touch point 1 of 3):
  After ingestion completes, calls bm25.mark_dirty() so BM25 index rebuilds
  on the next query. This handles the CLI path (python ingest_runner.py).
  The API path is handled separately in api.py BackgroundTask (touch point 3).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any

import modules.core as core
from modules import config
from modules.ingestion import loader, cleaner, chunker, embedder

logger = logging.getLogger("RAG-V3")


def run_ingestion(data_dir: Path = None) -> Dict[str, Any]:
    """
    Full offline ingestion pipeline: load → clean → chunk → embed → store.
    Returns: {files_processed, chunks_indexed, chunks_skipped, duration_ms, error}
    """
    data_dir = data_dir or config.DATA_DIR
    start = time.perf_counter()

    result: Dict[str, Any] = {
        "files_processed": 0,
        "chunks_indexed": 0,
        "chunks_skipped": 0,
        "duration_ms": 0,
        "error": None,
    }

    if not config.validate():
        result["error"] = "Environment validation failed — check .env file"
        return result

    try:
        # Step 1: Load
        docs = loader.load(data_dir)
        if not docs:
            result["error"] = f"No supported documents found in {data_dir}"
            return result

        result["files_processed"] = len(
            set(d.metadata.get("source", "") for d in docs)
        )

        # Step 2: Clean
        docs = cleaner.clean(docs)

        # Step 3: Chunk
        embeddings = core.get_embeddings()
        chunks = chunker.chunk(docs, embeddings)

        # Step 4: Embed & Store
        client = core.get_qdrant_client()
        embed_result = embedder.embed_and_store(chunks, client, embeddings)
        result["chunks_indexed"] = embed_result["chunks_indexed"]
        result["chunks_skipped"] = embed_result["chunks_skipped"]

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        result["error"] = str(e)

    finally:
        result["duration_ms"] = int((time.perf_counter() - start) * 1000)

    # ── Dirty-flag touch point 1/3 (CLI path) ─────────────────────────────────
    # Mark BM25 index dirty so it rebuilds on next query.
    # Safe to import here — bm25 module exists after Phase 4.
    try:
        from modules.retrieval import bm25
        bm25.mark_dirty()
        logger.info("BM25 marked dirty after ingestion")
    except ImportError:
        logger.debug("BM25 module not yet available — skipping mark_dirty (Phase 2 stub)")

    logger.info(
        f"Ingestion done: {result['files_processed']} file(s), "
        f"{result['chunks_indexed']} new chunks, "
        f"{result['chunks_skipped']} skipped, "
        f"{result['duration_ms']}ms"
    )
    return result
