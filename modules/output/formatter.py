"""
formatter.py — Output Formatting & Grounding Check
Responsibilities:
  1. format()         : Parse + validate LLM JSON response, enforce schema
  2. grounding_check(): Pure Python n-gram overlap check (zero LLM cost, zero latency)
                        Returns 0.0–1.0 grounding score used by pipeline.py
                        to gate the optional LLM hallucination guard call.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger("RAG-V3")

_FALLBACK: Dict[str, Any] = {
    "answer": "Information not found in the provided sources.",
    "sources": [],
    "confidence": 0.0,
}

_NGRAM_SIZE = 4  # 4-gram overlap for grounding check


# ── Output Formatting ──────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[str]:
    """Try to extract a valid JSON object from raw LLM output."""
    # Direct parse
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    stripped = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    stripped = re.sub(r'\s*```$', '', stripped.strip(), flags=re.MULTILINE)
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    # Find first {...} block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def format_response(raw: str) -> Dict[str, Any]:
    """
    Parse and validate LLM JSON response.
    Enforces schema: {answer: str, sources: list, confidence: float}
    Falls back to _FALLBACK if parsing fails or required fields are missing.
    """
    if not raw or not raw.strip():
        logger.warning("Formatter: empty LLM response — returning fallback")
        return dict(_FALLBACK)

    json_str = _extract_json(raw)
    if not json_str:
        logger.warning(f"Formatter: could not extract JSON from: {raw[:200]}")
        return dict(_FALLBACK)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Formatter: JSON parse error: {e}")
        return dict(_FALLBACK)

    # Validate required fields
    answer = data.get("answer", "")
    if not answer or not isinstance(answer, str) or not answer.strip():
        logger.warning("Formatter: missing or empty 'answer' field")
        return dict(_FALLBACK)

    return {
        "answer": answer.strip(),
        "sources": data.get("sources", []),
        "confidence": float(data.get("confidence", 0.5)),
    }


# ── Grounding Check (Stage 1 Hallucination Guard) ─────────────────────────────

def _get_ngrams(text: str, n: int) -> set:
    """Extract character n-grams from text (lowercased)."""
    text = text.lower()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def grounding_check(answer: str, context_chunks: List[str]) -> float:
    """
    Pure Python n-gram overlap check. Zero LLM calls. Zero latency cost.
    Splits the answer into sentences, checks what fraction of sentences
    have n-gram overlap with at least one context chunk.

    Returns: float 0.0–1.0
      1.0 = every sentence is grounded in context
      0.0 = no sentence overlaps with any context chunk
      
    Used by pipeline.py to decide whether to invoke the LLM hallucination guard.
    """
    if not answer or not context_chunks:
        return 0.0

    # Split answer into sentences (simple heuristic)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
    if not sentences:
        return 0.0

    # Build combined n-gram set from all context chunks
    context_ngrams: set = set()
    for chunk in context_chunks:
        context_ngrams |= _get_ngrams(chunk, _NGRAM_SIZE)

    if not context_ngrams:
        return 0.0

    grounded_count = 0
    for sentence in sentences:
        sentence_ngrams = _get_ngrams(sentence, _NGRAM_SIZE)
        if sentence_ngrams and sentence_ngrams & context_ngrams:
            grounded_count += 1

    score = grounded_count / len(sentences)
    logger.debug(
        f"Grounding check: {grounded_count}/{len(sentences)} sentences grounded "
        f"(score={score:.2f})"
    )
    return score


if __name__ == "__main__":
    context = ["The Q3 budget limit is $50,000 per department as per Section 4.2."]
    answer = "The Q3 budget limit is $50,000. However, there is no limit on executive spending."
    score = grounding_check(answer, context)
    print(f"Grounding score: {score:.2f}")  # Should be < 1.0 (second sentence not grounded)
