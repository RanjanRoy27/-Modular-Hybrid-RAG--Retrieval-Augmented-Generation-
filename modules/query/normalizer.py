"""
normalizer.py — Query Normalization Module
Responsibility: Validate and normalize raw query input.
Input : raw query string
Output: clean, safe query string — or raises ValueError
"""
from __future__ import annotations

import re
import logging

from modules import config

logger = logging.getLogger("RAG-V3")

_WHITESPACE = re.compile(r'\s+')


def normalize(query: str) -> str:
    """
    Normalize raw query string:
    - Strips leading/trailing whitespace
    - Collapses all internal whitespace (tabs, newlines, etc.) to single space
    - Validates not empty and not over MAX_QUESTION_LENGTH chars
    Raises ValueError on invalid input.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty")

    normalized = _WHITESPACE.sub(' ', query).strip()

    if len(normalized) > config.MAX_QUESTION_LENGTH:
        raise ValueError(
            f"Query too long ({len(normalized)} chars). "
            f"Max: {config.MAX_QUESTION_LENGTH}"
        )

    logger.debug(f"Query normalized: {repr(normalized[:80])}")
    return normalized


if __name__ == "__main__":
    for q in ["  What is  the   rent? ", "Normal query", ""]:
        try:
            print(f"OUT: {repr(normalize(q))}")
        except ValueError as e:
            print(f"ERR: {e}")
