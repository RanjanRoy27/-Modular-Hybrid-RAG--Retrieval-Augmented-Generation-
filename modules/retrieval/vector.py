"""
vector.py — Dense Vector Search Module
Responsibility: Semantic similarity search via Qdrant.
Supports optional metadata filtering (source_file, date range, etc.)
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from modules import config

logger = logging.getLogger("RAG-V3")


def search(
    query: str,
    embeddings,
    client: QdrantClient,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Dense vector search in Qdrant.
    Args:
        query   : normalized (and optionally HyDE-expanded) query
        filters : optional metadata filter dict, e.g. {"source_file": "policy.pdf"}
    Returns top TOP_K_RETRIEVAL documents with metadata.
    """
    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.COLLECTION_NAME,
            embedding=embeddings,
        )

        qdrant_filter = None
        if filters:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": config.TOP_K_RETRIEVAL,
                "filter": qdrant_filter,
            }
        )
        docs = retriever.invoke(query)
        logger.info(f"Vector search: {len(docs)} docs for '{query[:60]}'")
        return docs

    except Exception as e:
        logger.error(f"Vector search failed: {e}", exc_info=True)
        return []
