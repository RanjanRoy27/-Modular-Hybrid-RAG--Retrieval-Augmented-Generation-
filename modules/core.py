"""
core.py — Singleton Resource Initialization
All shared resources (LLM, embeddings, Qdrant client) are initialized once at startup.
Never call these per-request — that would reload models on every query.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from modules import config

logger = logging.getLogger("RAG-V3")

# ── Singletons ─────────────────────────────────────────────────────────────────
_llm: Optional[ChatGoogleGenerativeAI] = None
_embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
_qdrant_client: Optional[QdrantClient] = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            temperature=0,
            google_api_key=config.GOOGLE_API_KEY,
        )
        logger.info(f"LLM initialized: {config.LLM_MODEL}")
    return _llm


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBED_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )
        logger.info(f"Embeddings initialized: {config.EMBED_MODEL}")
    return _embeddings


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=config.QDRANT_PATH)
        _ensure_collection(_qdrant_client)
        logger.info(f"Qdrant client initialized: {config.QDRANT_PATH}")
    return _qdrant_client


def _ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it does not already exist."""
    if not client.collection_exists(config.COLLECTION_NAME):
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=config.VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection: {config.COLLECTION_NAME}")


def reset() -> None:
    """Reset all singletons (useful for testing)."""
    global _llm, _embeddings, _qdrant_client
    _llm = None
    _embeddings = None
    _qdrant_client = None
    logger.info("Core singletons reset")
