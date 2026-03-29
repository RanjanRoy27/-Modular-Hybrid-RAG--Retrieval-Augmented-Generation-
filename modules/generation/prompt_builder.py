"""
prompt_builder.py — Domain-Aware Prompt Construction
Responsibility: Build LLM message list with domain-specific instructions.
Enforces: no hallucination, only context, structured JSON output, fallback.
JSON examples in templates use {{ }} to avoid ChatPromptTemplate variable collision.
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger("RAG-V3")

# ── Base instructions (all domains) ────────────────────────────────────────────
_BASE = """\
You are a precise knowledge retrieval assistant.
Answer ONLY using the retrieved context provided below. Do NOT invent facts.
If the answer is not in the context, respond with exactly:
{{"answer": "Information not found in the provided sources.", "sources": [], "confidence": 0.0}}

Reference sources inline: e.g. "The limit is $50,000 [SOURCE 1]."
Output ONLY valid JSON — no markdown fences, no explanation — in this exact schema:
{{
  "answer": "Detailed answer with inline [SOURCE N] citations.",
  "sources": [
    {{"document": "filename.pdf", "page": "12", "excerpt": "exact quote used"}}
  ],
  "confidence": 0.85
}}"""

# ── Domain-specific prefixes (prepended to base) ────────────────────────────────
_PREFIXES = {
    "real_estate": (
        "This is a real estate query. Focus on lease terms, rent amounts, "
        "dates, obligations, and clause references. Extract specific figures.\n\n"
    ),
    "healthcare": (
        "This is a medical query. Be strictly precise about dosages, protocols, "
        "and diagnoses. Never extrapolate beyond what the source explicitly states. "
        "Reflect clinical uncertainty in the confidence score.\n\n"
    ),
    "generic": "",
}


def build(
    query: str,
    context: str,
    domain: str,
    history: List = None,
) -> List:
    """
    Build messages list for LLM invocation.
    Returns: [SystemMessage, ...history_turns..., HumanMessage(query)]
    Injects at most 6 history messages (3 Q/A pairs) to prevent context overflow.
    """
    history = history or []
    prefix = _PREFIXES.get(domain, "")
    system_content = (
        prefix
        + _BASE
        + f"\n\n{'─' * 40}\nRETRIEVED CONTEXT:\n{'─' * 40}\n{context}"
    )

    messages = [SystemMessage(content=system_content)]

    # Last 3 Q/A pairs (6 messages) from session history
    for turn in history[-6:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=query))
    logger.debug(
        f"Prompt: domain={domain}, context={len(context)} chars, "
        f"history={len(history)} turns, messages={len(messages)}"
    )
    return messages
