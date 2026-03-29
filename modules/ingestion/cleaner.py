"""
cleaner.py — Text Normalization Module
Responsibility: Remove noise from document text without losing semantic content.
Does NOT lowercase — LLMs are case-sensitive and proper nouns matter.
"""
from __future__ import annotations

import re
import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger("RAG-V3")

_PATTERNS = [
    (re.compile(r'\x00'),            ''),        # null bytes
    (re.compile(r'\r\n'),            '\n'),       # Windows line endings
    (re.compile(r'\r'),              '\n'),       # bare CR
    (re.compile(r'[ \t]+\n', re.M), '\n'),       # trailing whitespace before newline
    (re.compile(r'\n{3,}'),          '\n\n'),     # 3+ blank lines → 2
    (re.compile(r'[ \t]{2,}'),       ' '),        # multiple spaces/tabs → single space
]


def _clean_text(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


def clean(docs: List[Document]) -> List[Document]:
    """
    Clean and normalize text content of all documents.
    Filters out documents that become empty after cleaning.
    Metadata is preserved unchanged.
    """
    cleaned: List[Document] = []
    for doc in docs:
        text = _clean_text(doc.page_content)
        if not text:
            logger.debug(f"Removed empty doc from: {doc.metadata.get('source', 'unknown')}")
            continue
        cleaned.append(Document(page_content=text, metadata=doc.metadata))

    logger.info(f"Cleaner: {len(docs)} → {len(cleaned)} non-empty documents")
    return cleaned


if __name__ == "__main__":
    sample = Document(
        page_content="  Hello\r\n\r\n\r\n\r\nWorld  \t\n",
        metadata={"source": "test.txt"}
    )
    result = clean([sample])
    print(repr(result[0].page_content))
