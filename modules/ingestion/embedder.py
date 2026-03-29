"""
embedder.py — Embed & Store Module
Responsibility: Deduplicate by content_hash, then batch-embed and upsert to Qdrant.
Re-ingesting unchanged files is a no-op — only new/changed chunks are embedded.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from modules import config

logger = logging.getLogger("RAG-V3")

BATCH_SIZE = 100  # upsert in batches to avoid memory issues on large ingestions


def _get_existing_hashes(client: QdrantClient) -> set:
    """Scroll all existing content_hash values from Qdrant to enable deduplication."""
    existing: set = set()
    try:
        offset = None
        while True:
            results, next_offset = client.scroll(
                collection_name=config.COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=["content_hash"],
                with_vectors=False,
            )
            for point in results:
                h = (point.payload or {}).get("content_hash")
                if h:
                    existing.add(h)
            if next_offset is None:
                break
            offset = next_offset
    except Exception as e:
        logger.warning(f"Could not fetch existing hashes (collection may be empty): {e}")
    return existing


def embed_and_store(
    chunks: List[Document],
    client: QdrantClient,
    embeddings,
) -> Dict[str, Any]:
    """
    Deduplicate chunks by content_hash, then embed and upsert new chunks to Qdrant.
    Returns: {chunks_total, chunks_skipped, chunks_indexed}
    """
    if not chunks:
        return {"chunks_total": 0, "chunks_skipped": 0, "chunks_indexed": 0}

    logger.info(f"Checking {len(chunks)} chunks for duplicates...")
    existing_hashes = _get_existing_hashes(client)
    new_chunks = [
        c for c in chunks
        if c.metadata.get("content_hash") not in existing_hashes
    ]
    skipped = len(chunks) - len(new_chunks)
    logger.info(f"  {skipped} duplicate(s) skipped, {len(new_chunks)} new chunk(s) to embed")

    if not new_chunks:
        return {"chunks_total": len(chunks), "chunks_skipped": skipped, "chunks_indexed": 0}

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding=embeddings,
    )

    total_indexed = 0
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i: i + BATCH_SIZE]
        vector_store.add_documents(batch)
        total_indexed += len(batch)
        logger.info(f"  Upserted batch {i // BATCH_SIZE + 1}: {len(batch)} chunks")

    logger.info(f"Embedding complete — {total_indexed} chunks indexed.")
    return {
        "chunks_total": len(chunks),
        "chunks_skipped": skipped,
        "chunks_indexed": total_indexed,
    }
