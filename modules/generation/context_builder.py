"""
context_builder.py — Context Assembly Module
Responsibility: Assemble reranked docs into LLM context string + raw chunk list.

Returns tuple[str, List[str]]:
  - context_str : formatted string injected into LLM prompt
  - raw_chunks  : plain text list used by grounding_check() in formatter.py
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger("RAG-V3")

MAX_CONTEXT_CHARS = 12_000  # Stay well under Gemini context limit for efficiency


def build(docs: List[Document]) -> Tuple[str, List[str]]:
    """
    Build context string and raw chunk list from reranked documents.

    Returns:
        (context_str, raw_chunks)
        context_str : citation-tagged string for LLM injection
        raw_chunks  : plain text list for grounding_check()
    """
    if not docs:
        return "", []

    formatted_parts: List[str] = []
    raw_chunks: List[str] = []
    total_chars = 0

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page_number", doc.metadata.get("page", "N/A"))
        content = doc.page_content.strip()

        raw_chunks.append(content)
        entry = f"[SOURCE {i}: {source} | PAGE: {page}]\n{content}"

        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                entry = entry[:remaining] + "\n[... truncated]"
                formatted_parts.append(entry)
            logger.info(f"Context limit reached at doc {i}/{len(docs)} — truncated")
            break

        formatted_parts.append(entry)
        total_chars += len(entry)

    context_str = "\n\n".join(formatted_parts)
    logger.info(f"Context built: {len(formatted_parts)} chunks, ~{total_chars} chars")
    return context_str, raw_chunks
