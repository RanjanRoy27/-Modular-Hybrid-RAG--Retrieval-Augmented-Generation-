"""
detector.py — Domain Detection Module
Responsibility: Classify user query into a domain for domain-aware prompt selection.
Uses keyword heuristics first (fast, zero token cost), LLM fallback only if ambiguous.
Output: 'real_estate' | 'healthcare' | 'generic'
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("RAG-V3")

DOMAIN_REAL_ESTATE = "real_estate"
DOMAIN_HEALTHCARE = "healthcare"
DOMAIN_GENERIC = "generic"

_REAL_ESTATE_KW = {
    "rent", "lease", "tenant", "landlord", "property", "mortgage",
    "eviction", "zoning", "building", "apartment", "unit", "listing",
    "real estate", "title deed", "escrow", "sq ft", "square feet",
    "sublease", "foreclosure", "lien", "easement", "hoa",
}

_HEALTHCARE_KW = {
    "patient", "treatment", "diagnosis", "medication", "drug", "dosage",
    "clinical", "hospital", "surgery", "symptom", "prescription", "protocol",
    "antibiotic", "therapy", "physician", "nurse", "discharge", "icu",
    "medical", "health", "disease", "condition", "allergy", "lab result",
}


def _keyword_detect(query: str) -> Optional[str]:
    """Fast keyword scoring — returns None if ambiguous (scores tie or both zero)."""
    lower = query.lower()
    re_score = sum(1 for kw in _REAL_ESTATE_KW if kw in lower)
    h_score = sum(1 for kw in _HEALTHCARE_KW if kw in lower)

    if re_score > h_score and re_score >= 1:
        return DOMAIN_REAL_ESTATE
    if h_score > re_score and h_score >= 1:
        return DOMAIN_HEALTHCARE
    return None  # ambiguous


def detect(query: str, llm=None) -> str:
    """
    Detect query domain.
    1. Keyword heuristics (fast path)
    2. LLM classification (fallback, only if llm provided and heuristics fail)
    3. Default to 'generic'
    """
    domain = _keyword_detect(query)
    if domain is not None:
        logger.debug(f"Domain (heuristic): {domain}")
        return domain

    if llm is not None:
        try:
            from langchain_core.messages import HumanMessage
            prompt = (
                "Classify this query into exactly one of: "
                "'real_estate', 'healthcare', or 'generic'. "
                "Reply with ONLY the label.\n\nQuery: " + query
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            detected = resp.content.strip().lower()
            if detected in (DOMAIN_REAL_ESTATE, DOMAIN_HEALTHCARE, DOMAIN_GENERIC):
                logger.info(f"Domain (LLM): {detected}")
                return detected
        except Exception as e:
            logger.warning(f"Domain LLM fallback failed: {e}")

    logger.debug("Domain: defaulting to generic")
    return DOMAIN_GENERIC


if __name__ == "__main__":
    tests = [
        "What is the rent for the downtown apartment?",
        "What is the recommended dosage for amoxicillin?",
        "What are the Q3 budget limits?",
    ]
    for q in tests:
        print(f"'{q}' → {detect(q)}")
