"""
loader.py — Document Loading Module
Responsibility: Load raw files from data/ into LangChain Documents.
Input : data_dir: Path
Output: List[Document] with source + page_number metadata
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger("RAG-V3")

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".xlsx", ".xls"}


def load(data_dir: Path) -> List[Document]:
    """
    Scan data_dir and load all supported documents.
    Stamps source and page_number on every Document metadata.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = [
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        logger.warning(f"No supported files found in {data_dir}")
        return []

    logger.info(f"Found {len(files)} file(s) in {data_dir}")
    all_docs: List[Document] = []
    counts = {"txt": 0, "pdf": 0, "docx": 0, "xlsx": 0, "skipped": 0}

    for filepath in files:
        ext = filepath.suffix.lower()
        try:
            if ext == ".txt":
                loader = TextLoader(str(filepath), encoding="utf-8")
                docs = loader.load()
                counts["txt"] += 1
            elif ext == ".pdf":
                loader = PyPDFLoader(str(filepath))
                docs = loader.load()
                counts["pdf"] += 1
            elif ext == ".docx":
                loader = Docx2txtLoader(str(filepath))
                docs = loader.load()
                counts["docx"] += 1
            elif ext in (".xlsx", ".xls"):
                loader = UnstructuredExcelLoader(str(filepath), mode="elements")
                docs = loader.load()
                counts["xlsx"] += 1
            else:
                counts["skipped"] += 1
                continue

            for doc in docs:
                doc.metadata["source"] = filepath.name
                doc.metadata.setdefault(
                    "page_number", doc.metadata.get("page", 0)
                )

            all_docs.extend(docs)
            logger.info(f"  [OK] {filepath.name} — {len(docs)} page(s)")

        except Exception as e:
            logger.error(f"  [FAIL] {filepath.name} — {e}")
            counts["skipped"] += 1

    logger.info(
        f"Loaded: {counts['txt']} TXT | {counts['pdf']} PDF | "
        f"{counts['docx']} DOCX | {counts['xlsx']} XLSX | {counts['skipped']} skipped"
    )
    return all_docs


if __name__ == "__main__":
    from modules.config import DATA_DIR
    docs = load(DATA_DIR)
    print(f"Total docs loaded: {len(docs)}")
