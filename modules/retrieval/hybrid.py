"""
hybrid.py — Hybrid Retrieval Merger
Responsibility: Merge vector and BM25 results using Reciprocal Rank Fusion (RRF).
Deduplicates by content_hash, normalizes scores across both retrieval paths.

RRF formula: score(d) = sum over lists: 1 / (k + rank(d))
Standard k=60 is used — balances high-rank and low-rank documents.
"""
from __future__ import annotations

import logging
from typing import List, Dict

from langchain_core.documents import Document

logger = logging.getLogger("RAG-V3")

RRF_K = 60  # Standard RRF constant


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def _doc_key(doc: Document) -> str:
    """Deduplication key: prefer content_hash, fallback to first 100 chars."""
    return doc.metadata.get("content_hash") or doc.page_content[:100]


def merge(
    vector_docs: List[Document],
    bm25_docs: List[Document],
    max_results: int = 30,
) -> List[Document]:
    """
    Merge vector and BM25 results using Reciprocal Rank Fusion.
    Deduplicates across both lists. Aggregates RRF scores additively.
    Returns top max_results documents sorted by combined RRF score.
    """
    if not vector_docs and not bm25_docs:
        return []

    combined: Dict[str, Dict] = {}  # key → {doc, rrf_score}

    for rank, doc in enumerate(vector_docs):
        key = _doc_key(doc)
        score = _rrf_score(rank)
        if key in combined:
            combined[key]["rrf_score"] += score
        else:
            combined[key] = {"doc": doc, "rrf_score": score}

    for rank, doc in enumerate(bm25_docs):
        key = _doc_key(doc)
        score = _rrf_score(rank)
        if key in combined:
            combined[key]["rrf_score"] += score
            # Carry bm25_score into merged doc metadata
            combined[key]["doc"].metadata["bm25_score"] = doc.metadata.get("bm25_score", 0)
        else:
            combined[key] = {"doc": doc, "rrf_score": score}

    sorted_items = sorted(
        combined.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    results: List[Document] = []
    for item in sorted_items[:max_results]:
        doc = item["doc"]
        doc.metadata["rrf_score"] = round(item["rrf_score"], 6)
        results.append(doc)

    logger.info(
        f"Hybrid merge: {len(vector_docs)} vector + {len(bm25_docs)} BM25 "
        f"→ {len(results)} merged (deduped, RRF scored)"
    )
    return results


if __name__ == "__main__":
    v = [Document(page_content=f"Vector {i}", metadata={"content_hash": f"v{i}"}) for i in range(5)]
    b = [Document(page_content=f"BM25 {i}",   metadata={"content_hash": f"b{i}"}) for i in range(5)]
    # Overlap: share hash b2=v3
    b[2].metadata["content_hash"] = "v3"
    merged = merge(v, b)
    print(f"Merged: {len(merged)} docs")
    for d in merged:
        print(f"  rrf={d.metadata['rrf_score']:.5f} | {d.page_content}")
