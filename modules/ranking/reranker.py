"""
reranker.py — Cross-Encoder Reranker
Responsibility: Re-score hybrid-merged docs for precision.
Input : top 20-30 hybrid docs
Output: top TOP_K_RERANK docs — only the best go to the LLM.

Why cross-encoder vs cosine: cosine measures geometric proximity in embedding space.
Cross-encoders read (query, document) jointly and capture linguistic interaction,
giving 15-30% precision improvement over bi-encoder retrieval alone.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

from modules import config

logger = logging.getLogger("RAG-V3")

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder: {config.RERANKER_MODEL}")
        _model = CrossEncoder(config.RERANKER_MODEL)
        logger.info("Cross-encoder loaded successfully")
    return _model


def rerank(
    query: str,
    docs: List[Document],
    top_k: Optional[int] = None,
) -> List[Document]:
    """
    Re-rank documents using cross-encoder scoring.
    Preserves all original metadata. Adds rerank_score to each doc's metadata.
    Falls back to returning first top_k docs if model fails.
    """
    top_k = top_k or config.TOP_K_RERANK

    if not docs:
        return []

    try:
        model = _get_model()
        pairs = [[query, doc.page_content] for doc in docs]
        scores = model.predict(pairs)

        for i, doc in enumerate(docs):
            doc.metadata["rerank_score"] = float(scores[i])

        reranked = sorted(
            docs,
            key=lambda d: d.metadata.get("rerank_score", 0.0),
            reverse=True,
        )
        top = reranked[:top_k]

        logger.info(
            f"Reranker: {len(docs)} → {len(top)} docs | "
            f"top score: {top[0].metadata.get('rerank_score', 0):.4f}"
        )
        return top

    except Exception as e:
        logger.error(f"Reranker failed, falling back to unsorted: {e}", exc_info=True)
        return docs[:top_k]
