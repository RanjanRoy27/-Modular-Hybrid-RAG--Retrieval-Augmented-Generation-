"""
logger.py — Structured Observability Module
Responsibility: Emit one JSON line per query to rag_trace.log.
Append-only. Never blocks the query path.
Fields include grounding_score and hallucination_warning (new in V3).
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Any

from modules import config

logger = logging.getLogger("RAG-V3")

_trace_logger: Optional[logging.Logger] = None


def _get_trace_logger() -> logging.Logger:
    global _trace_logger
    if _trace_logger is None:
        _trace_logger = logging.getLogger("RAG-TRACE")
        _trace_logger.setLevel(logging.INFO)
        _trace_logger.propagate = False  # Don't pollute the root logger

        handler = logging.FileHandler(str(config.LOG_FILE), encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        _trace_logger.addHandler(handler)
    return _trace_logger


def new_query_id() -> str:
    """Generate a unique query ID (UUID4)."""
    return str(uuid.uuid4())


def log_query(
    query_id: str,
    query: str,
    domain: str,
    chunks_retrieved: int,
    rerank_scores: List[float],
    retrieval_ms: int,
    llm_ms: int,
    total_ms: int,
    fallback_triggered: bool,
    grounding_score: float,
    hallucination_warning: bool,
    prompt_tokens: Optional[int] = None,
    hyde_used: bool = False,
    guard_triggered: bool = False,
) -> None:
    """
    Emit one structured JSON line to rag_trace.log.
    Non-blocking — if write fails, logs a warning and continues.
    """
    record = {
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_preview": query[:120],
        "domain": domain,
        "hyde_used": hyde_used,
        "chunks_retrieved": chunks_retrieved,
        "rerank_scores": [round(s, 4) for s in rerank_scores],
        "prompt_tokens": prompt_tokens,
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
        "total_ms": total_ms,
        "fallback_triggered": fallback_triggered,
        "grounding_score": round(grounding_score, 4),
        "hallucination_warning": hallucination_warning,
        "guard_triggered": guard_triggered,
    }

    try:
        _get_trace_logger().info(json.dumps(record, ensure_ascii=False))
    except Exception as e:
        logger.warning(f"Trace log write failed: {e}")


if __name__ == "__main__":
    qid = new_query_id()
    log_query(
        query_id=qid,
        query="What is the Q3 budget limit?",
        domain="generic",
        chunks_retrieved=5,
        rerank_scores=[0.92, 0.87, 0.74, 0.61, 0.55],
        retrieval_ms=120,
        llm_ms=840,
        total_ms=980,
        fallback_triggered=False,
        grounding_score=0.88,
        hallucination_warning=False,
    )
    print(f"Logged trace for query_id: {qid}")
    print(f"Check: {config.LOG_FILE}")
