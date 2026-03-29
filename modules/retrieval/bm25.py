"""
bm25.py — BM25 Keyword Search Module
Responsibility: In-memory BM25 keyword search with dirty-flag pattern.

Dirty-flag wiring — 3 touch points:
  1. HERE (Phase 4) : _dirty flag, build(), mark_dirty() implemented
  2. ingestion/pipeline.py (Phase 2, wired Phase 4): calls mark_dirty() at end of run_ingestion()
  3. api.py (Phase 9): BackgroundTask completion hook calls mark_dirty()

No disk persistence. No pickle. No joblib. Index lives in memory only.
Rebuilt from Qdrant whenever _dirty=True and search() is called.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger("RAG-V3")

# ── Module-level state (dirty-flag pattern) ────────────────────────────────────
_index = None                   # BM25Okapi instance (None = not yet built)
_corpus: List[List[str]] = []   # tokenized corpus aligned with _documents
_documents: List[Document] = [] # original Document objects
_dirty: bool = True             # True = must rebuild before next search


def mark_dirty() -> None:
    """
    Mark BM25 index as stale.
    Call this after any ingestion (CLI path via ingestion/pipeline.py,
    API path via api.py BackgroundTask completion hook).
    Safe to call from any context.
    """
    global _dirty
    _dirty = True
    logger.info("BM25 index marked dirty — will rebuild on next search call")


def _tokenize(text: str) -> List[str]:
    """Whitespace tokenizer for BM25 scoring."""
    return text.lower().split()


def build(client=None) -> None:
    """
    (Re)build BM25 index from all chunks in Qdrant.
    Scrolls entire collection, tokenizes text, builds BM25Okapi.
    Sets _dirty = False after successful build.
    """
    global _index, _corpus, _documents, _dirty

    try:
        from rank_bm25 import BM25Okapi
        from modules import config

        if client is None:
            import modules.core as core
            client = core.get_qdrant_client()

        logger.info("Building BM25 index from Qdrant...")
        all_docs: List[Document] = []
        offset = None

        while True:
            results, next_offset = client.scroll(
                collection_name=config.COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                payload = point.payload or {}
                content = payload.get("page_content", payload.get("content", ""))
                if content:
                    all_docs.append(Document(page_content=content, metadata=payload))
            if next_offset is None:
                break
            offset = next_offset

        if not all_docs:
            logger.warning("BM25 build: Qdrant is empty — index not built")
            _index = None
            _corpus = []
            _documents = []
            _dirty = False
            return

        _documents = all_docs
        _corpus = [_tokenize(doc.page_content) for doc in all_docs]
        _index = BM25Okapi(_corpus)
        _dirty = False
        logger.info(f"BM25 index built: {len(_documents)} documents indexed")

    except ImportError:
        logger.error("rank-bm25 not installed. Run: pip install rank-bm25")
        _dirty = False  # prevent retry on every query
    except Exception as e:
        logger.error(f"BM25 build failed: {e}", exc_info=True)
        _dirty = False


def search(query: str, top_k: Optional[int] = None) -> List[Document]:
    """
    BM25 keyword search.
    Auto-rebuilds index if _dirty=True before serving results.
    Returns top_k Documents with bm25_score in metadata.
    Only returns docs with score > 0 (exact matches only).
    """
    global _dirty, _index

    from modules import config
    top_k = top_k or config.TOP_K_RETRIEVAL

    # Auto-rebuild if dirty (lazy initialization pattern)
    if _dirty or _index is None:
        build()

    if _index is None or not _documents:
        logger.warning("BM25 search: no index available — returning empty")
        return []

    try:
        tokenized = _tokenize(query)
        scores = _index.get_scores(tokenized)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )
        top_indices = ranked[:top_k]

        results: List[Document] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break  # no more matches
            doc = Document(
                page_content=_documents[idx].page_content,
                metadata={**_documents[idx].metadata, "bm25_score": float(scores[idx])},
            )
            results.append(doc)

        logger.info(f"BM25 search: {len(results)} docs with score > 0 for '{query[:60]}'")
        return results

    except Exception as e:
        logger.error(f"BM25 search failed: {e}", exc_info=True)
        return []


if __name__ == "__main__":
    mark_dirty()
    build()
    results = search("Q3 budget limit", top_k=3)
    print(f"BM25 results: {len(results)}")
    for r in results:
        print(f"  score={r.metadata.get('bm25_score', 0):.4f} | {r.page_content[:80]}")
