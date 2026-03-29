"""
expander.py — HyDE Query Expansion Module
Responsibility: Generate hypothetical answer embedding for short, vague queries.
Bypass regex checked FIRST — structured queries skip HyDE regardless of word count.

Logic order in expand():
  1. Bypass pattern check → return original if match (FIRST PRIORITY)
  2. Word count check     → return original if >= HYDE_THRESHOLD_WORDS
  3. Run HyDE LLM call   → return expanded query (only if both checks pass)
"""
from __future__ import annotations

import re
import logging
from typing import Optional

from langchain_core.messages import HumanMessage

from modules import config

logger = logging.getLogger("RAG-V3")

# Structured-query bypass: these patterns destroy retrieval precision if expanded
# Matches: "4.2", "18%", "GST", "FDA", "Section 4", "clause 7", "article 12", "page 5"
HYDE_BYPASS = re.compile(
    r'\d+\.\d+|'           # decimal numbers: 4.2, 18.5
    r'\d+\s*%|'            # percentages: 18%, 5 %
    r'\b[A-Z]{2,}\b|'      # all-caps acronyms: GST, FDA, HIPAA
    r'section\s+\d+|'      # "section 4"
    r'clause\s+\d+|'       # "clause 7"
    r'article\s+\d+|'      # "article 12"
    r'page\s+\d+',         # "page 5"
    re.IGNORECASE,
)

_HYDE_PROMPT = (
    "Generate a concise, factual passage that would directly answer the following "
    "question if it appeared in a business document. "
    "Output ONLY the passage — no preamble, no explanation:\n\nQuestion: {query}"
)


def expand(query: str, llm) -> str:
    """
    Optionally expand query using HyDE.

    Returns original query unchanged if:
      - Query matches HYDE_BYPASS pattern (structured lookup)
      - Query word count >= HYDE_THRESHOLD_WORDS
      - HyDE LLM call fails (graceful fallback)

    Returns expanded hypothetical passage if HyDE runs successfully.
    """
    # 1. Bypass pattern check — FIRST (before word count)
    if HYDE_BYPASS.search(query):
        logger.debug(f"HyDE bypassed (structured query): {repr(query[:60])}")
        return query

    # 2. Word count check
    word_count = len(query.split())
    if word_count >= config.HYDE_THRESHOLD_WORDS:
        logger.debug(
            f"HyDE skipped ({word_count} words >= threshold {config.HYDE_THRESHOLD_WORDS})"
        )
        return query

    # 3. Run HyDE
    logger.info(f"HyDE expanding ({word_count}-word query): {repr(query[:60])}")
    try:
        prompt = _HYDE_PROMPT.format(query=query)
        response = llm.invoke([HumanMessage(content=prompt)])
        expanded = response.content.strip()
        if expanded:
            logger.info(f"HyDE result: {repr(expanded[:100])}")
            return expanded
    except Exception as e:
        logger.warning(f"HyDE expansion failed — using original query: {e}")

    return query


if __name__ == "__main__":
    test_cases = [
        ("Section 4.2", "bypass — structured ref"),
        ("GST 18%", "bypass — acronym + decimal"),
        ("clause 7B", "bypass — clause reference"),
        ("budget", "should expand — 1 word"),
        ("What is the budget limit for Q3 procurement requests?", "skip — long query"),
    ]
    for query, note in test_cases:
        bypass = bool(HYDE_BYPASS.search(query))
        wc = len(query.split())
        long = wc >= config.HYDE_THRESHOLD_WORDS
        action = "BYPASS" if bypass else ("SKIP (long)" if long else "WOULD EXPAND")
        print(f"[{action}] {repr(query)} — {note}")
