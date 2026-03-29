"""
chunker.py — Semantic Chunking Module
Responsibility: Split documents at semantic boundaries and inject rich metadata.
Uses SemanticChunker (meaning-aware) instead of fixed character splits.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

logger = logging.getLogger("RAG-V3")


def chunk(docs: List[Document], embeddings) -> List[Document]:
    """
    Split documents using SemanticChunker, then inject per-chunk metadata:
      - source_file       : original filename
      - page_number       : page from loader (or 0)
      - chunk_index       : global position in output list
      - ingestion_timestamp: ISO 8601 UTC
      - content_hash      : SHA-256 of chunk text (used for deduplication)
      - section_heading   : first 80 chars of chunk (heuristic heading)
    """
    if not docs:
        return []

    logger.info(f"SemanticChunker: splitting {len(docs)} documents...")
    splitter = SemanticChunker(embeddings)
    chunks = splitter.split_documents(docs)

    timestamp = datetime.now(timezone.utc).isoformat()

    for i, c in enumerate(chunks):
        c.metadata["source_file"] = c.metadata.get("source", "unknown")
        c.metadata["page_number"] = c.metadata.get(
            "page_number", c.metadata.get("page", 0)
        )
        c.metadata["chunk_index"] = i
        c.metadata["ingestion_timestamp"] = timestamp
        c.metadata["content_hash"] = hashlib.sha256(
            c.page_content.encode("utf-8")
        ).hexdigest()
        first_line = c.page_content.strip().split("\n")[0][:80]
        c.metadata["section_heading"] = first_line or "Unknown"

    logger.info(f"SemanticChunker: {len(docs)} docs → {len(chunks)} chunks")
    return chunks
