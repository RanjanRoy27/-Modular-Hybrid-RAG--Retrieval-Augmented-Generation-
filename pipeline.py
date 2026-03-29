"""
pipeline.py — End-to-End Query Pipeline Orchestrator
Wires all modules in strict linear flow per the V3 spec.

⚠️  Dirty-flag touch point 2/3:
    pipeline.py does NOT call bm25.mark_dirty().
    mark_dirty() is called ONLY by:
      - ingestion/pipeline.py  → CLI path
      - api.py BackgroundTask  → API path (touch point 3)

Flow:
  normalize → expand → detect_domain
  → vector_search + bm25_search → hybrid_merge
  → rerank → build_context → build_prompt → llm_generate
  → format_output → grounding_check → [LLM guard if domain-gated + low score]
  → log_trace → return response dict
"""
from __future__ import annotations

import logging
import time
from typing import List, Dict, Any, Optional

import modules.core as core
from modules import config
from modules.query import normalizer, expander
from modules.domain import detector
from modules.retrieval import vector, bm25, hybrid
from modules.ranking import reranker
from modules.generation import context_builder, prompt_builder, llm as llm_module
from modules.output import formatter
from modules.observability import logger as obs_logger

logger = logging.getLogger("RAG-V3")

_GUARD_PROMPT = (
    "Given these source chunks:\n{context}\n\n"
    "Does the following answer contain ANY claims NOT supported by the sources above? "
    "Reply with only YES or NO, then a brief reason.\n\nAnswer: {answer}"
)


def run(
    question: str,
    session_id: str = "default",
    history: Optional[List] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute the full V3 query pipeline.

    Returns:
        {
          answer, sources, confidence,
          query_id, domain, latency_ms,
          grounding_score, hallucination_warning,
          fallback_triggered
        }
    """
    history = history or []
    filters = filters or {}
    query_id = obs_logger.new_query_id()
    total_start = time.perf_counter()

    # ── Step 1: Normalize ──────────────────────────────────────────────────────
    try:
        query = normalizer.normalize(question)
    except ValueError as e:
        return _error_response(query_id, str(e))

    # ── Step 2: Domain Detection ───────────────────────────────────────────────
    llm = core.get_llm()
    domain = detector.detect(query, llm=llm)

    # ── Step 3: HyDE Expansion ────────────────────────────────────────────────
    retrieval_start = time.perf_counter()
    expanded_query = expander.expand(query, llm)
    hyde_used = expanded_query != query

    # ── Step 4: Hybrid Retrieval ───────────────────────────────────────────────
    embeddings = core.get_embeddings()
    client = core.get_qdrant_client()

    vector_docs = vector.search(expanded_query, embeddings, client, filters or None)
    bm25_docs = bm25.search(expanded_query)
    merged_docs = hybrid.merge(vector_docs, bm25_docs)

    # ── Step 5: Rerank ────────────────────────────────────────────────────────
    top_docs = reranker.rerank(query, merged_docs)
    retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)

    fallback_triggered = len(top_docs) == 0
    if fallback_triggered:
        logger.warning(f"[{query_id}] No documents retrieved — returning fallback")
        return _fallback_response(query_id, domain, retrieval_ms)

    # ── Step 6: Build Context ─────────────────────────────────────────────────
    context_str, raw_chunks = context_builder.build(top_docs)

    # ── Step 7: Build Prompt ──────────────────────────────────────────────────
    messages = prompt_builder.build(query, context_str, domain, history)

    # ── Step 8: Generate ──────────────────────────────────────────────────────
    llm_start = time.perf_counter()
    try:
        raw_response = llm_module.generate(messages, llm)
    except Exception as e:
        logger.error(f"[{query_id}] LLM generation failed: {e}")
        return _error_response(query_id, f"LLM error: {e}")
    llm_ms = int((time.perf_counter() - llm_start) * 1000)

    # ── Step 9: Format Output ─────────────────────────────────────────────────
    response = formatter.format_response(raw_response)

    # ── Step 10: Two-Stage Hallucination Guard ────────────────────────────────
    grounding_score = formatter.grounding_check(response["answer"], raw_chunks)
    hallucination_warning = False
    guard_triggered = False

    # Stage 1: Python grounding check (zero cost — always runs)
    if grounding_score < config.GROUNDING_THRESHOLD:
        hallucination_warning = True
        logger.warning(
            f"[{query_id}] Low grounding score: {grounding_score:.2f} "
            f"(threshold: {config.GROUNDING_THRESHOLD})"
        )

        # Stage 2: LLM guard — only if domain is sensitive
        if domain in config.HALLUCINATION_GUARD_DOMAINS:
            guard_triggered = True
            logger.info(f"[{query_id}] Triggering LLM hallucination guard (domain: {domain})")
            try:
                guard_prompt_text = _GUARD_PROMPT.format(
                    context=context_str[:3000],
                    answer=response["answer"],
                )
                from langchain_core.messages import HumanMessage
                guard_resp = llm.invoke([HumanMessage(content=guard_prompt_text)])
                guard_result = guard_resp.content.strip()
                logger.info(f"[{query_id}] Guard response: {guard_result[:100]}")
            except Exception as e:
                logger.warning(f"[{query_id}] LLM guard call failed: {e}")

    # ── Step 11: Log Trace ────────────────────────────────────────────────────
    rerank_scores = [
        d.metadata.get("rerank_score", 0.0) for d in top_docs
    ]
    total_ms = int((time.perf_counter() - total_start) * 1000)

    obs_logger.log_query(
        query_id=query_id,
        query=query,
        domain=domain,
        chunks_retrieved=len(top_docs),
        rerank_scores=rerank_scores,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        total_ms=total_ms,
        fallback_triggered=fallback_triggered,
        grounding_score=grounding_score,
        hallucination_warning=hallucination_warning,
        hyde_used=hyde_used,
        guard_triggered=guard_triggered,
    )

    return {
        "answer": response["answer"],
        "sources": response["sources"],
        "confidence": response["confidence"],
        "query_id": query_id,
        "domain": domain,
        "latency_ms": total_ms,
        "grounding_score": grounding_score,
        "hallucination_warning": hallucination_warning,
        "fallback_triggered": fallback_triggered,
    }


def _fallback_response(query_id: str, domain: str, retrieval_ms: int) -> Dict[str, Any]:
    return {
        "answer": "No relevant information found in the knowledge base.",
        "sources": [],
        "confidence": 0.0,
        "query_id": query_id,
        "domain": domain,
        "latency_ms": retrieval_ms,
        "grounding_score": 0.0,
        "hallucination_warning": False,
        "fallback_triggered": True,
    }


def _error_response(query_id: str, error: str) -> Dict[str, Any]:
    return {
        "answer": f"An error occurred: {error}",
        "sources": [],
        "confidence": 0.0,
        "query_id": query_id,
        "domain": "generic",
        "latency_ms": 0,
        "grounding_score": 0.0,
        "hallucination_warning": False,
        "fallback_triggered": True,
        "error": error,
    }
