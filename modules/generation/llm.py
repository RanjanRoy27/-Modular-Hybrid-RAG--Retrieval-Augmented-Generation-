"""
llm.py — LLM Generation Wrapper
Responsibility: Single LLM call with exponential-backoff retry on 429/quota errors.
Returns raw string content — formatting/parsing is done by formatter.py.
"""
from __future__ import annotations

import logging
import time
from typing import List

logger = logging.getLogger("RAG-V3")

_MAX_RETRIES = 3
_BACKOFF_SECONDS = [2, 5, 10]


def generate(messages: List, llm) -> str:
    """
    Invoke LLM with retry on rate-limit errors.
    Returns raw response content string.
    Raises RuntimeError after all retries exhausted.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            response = llm.invoke(messages)
            content = response.content

            # Gemini can return list-structured content
            if isinstance(content, list):
                content = " ".join(
                    part["text"]
                    for part in content
                    if isinstance(part, dict) and "text" in part
                )

            logger.debug(f"LLM responded: {len(content)} chars")
            return content.strip()

        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(kw in err for kw in ("429", "quota", "rate limit", "resource exhausted"))

            if is_rate_limit and attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_SECONDS[attempt]
                logger.warning(
                    f"Rate limit (attempt {attempt + 1}/{_MAX_RETRIES}). "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
                continue

            logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
            raise

    raise RuntimeError(f"LLM generation failed after {_MAX_RETRIES} attempts")
